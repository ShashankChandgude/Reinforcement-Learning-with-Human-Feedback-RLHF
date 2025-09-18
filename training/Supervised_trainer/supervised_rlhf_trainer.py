"""
Supervised RLHF Trainer - Stable alternative to PPO
Uses reward model to rank responses and trains on preferred examples.
"""

import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers.trainer_utils import EvalPrediction
from torch.utils.data import Dataset
from data.data_loader import load_dataset
from training.Reward_Model.reward_model import RewardModel
from utils.logging_utils import setup_logger
from typing import Dict, Any, List, Tuple
import json
import numpy as np


class RankedResponseDataset(Dataset):
    """Dataset that ranks responses using reward model and keeps only preferred ones."""
    
    def __init__(self, data, reward_model, tokenizer, max_length=224, device='cuda'):
        self.data = data
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.ranked_data = []
        
        self.logger = setup_logger("ranked_dataset")
        self.logger.info("Ranking responses using reward model...")
        
        self._rank_responses()
        
    def _rank_responses(self):
        """Rank responses and keep only preferred ones for supervised training."""
        for item in self.data:
            prompt = item['prompt']
            chosen = item['chosen_text']
            rejected = item['rejected_text']
            
            # Score both responses
            chosen_score = self._score_response(prompt, chosen)
            rejected_score = self._score_response(prompt, rejected)
            
            # Keep the better response for supervised training
            if chosen_score > rejected_score:
                best_response = chosen
                score_diff = chosen_score - rejected_score
            else:
                best_response = rejected
                score_diff = rejected_score - chosen_score
            
            # Only include examples where the preference is clear (score difference > threshold)
            if score_diff > 0.1:  # Threshold for clear preference
                full_text = prompt + best_response
                
                # Tokenize
                encoded = self.tokenizer(
                    full_text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                self.ranked_data.append({
                    'input_ids': encoded['input_ids'].squeeze(),
                    'attention_mask': encoded['attention_mask'].squeeze(),
                    'labels': encoded['input_ids'].squeeze().clone(),  # For causal LM
                    'score': max(chosen_score, rejected_score),
                    'score_diff': score_diff
                })
        
        self.logger.info(f"Ranked {len(self.data)} pairs -> {len(self.ranked_data)} high-quality examples")
    
    def _score_response(self, prompt: str, response: str) -> float:
        """Score a response using the reward model."""
        full_text = prompt + response
        
        # Tokenize
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        # Get reward score
        with torch.no_grad():
            reward = self.reward_model(
                encoded['input_ids'], 
                attention_mask=encoded['attention_mask']
            )
        
        return reward.item()
    
    def __len__(self):
        return len(self.ranked_data)
    
    def __getitem__(self, idx):
        return self.ranked_data[idx]


class SupervisedRLHFTrainer:
    """Supervised RLHF trainer using reward-ranked responses."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logger("supervised_rlhf")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load base model
        model_name = config["model"]
        self.logger.info(f"Loading base model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Load reward model
        reward_model_dir = config.get("reward_model_dir")
        if reward_model_dir and os.path.exists(reward_model_dir):
            self.logger.info(f"Loading reward model from: {reward_model_dir}")
            base_rm = AutoModelForCausalLM.from_pretrained(reward_model_dir)
            self.reward_model = RewardModel(base_rm).to(self.device)
            
            # Load reward model weights
            reward_weights_path = os.path.join(reward_model_dir, "reward_model.pth")
            if os.path.exists(reward_weights_path):
                reward_state = torch.load(reward_weights_path, map_location=self.device)
                self.reward_model.load_state_dict(reward_state)
                self.logger.info("Reward model weights loaded")
        else:
            raise ValueError(f"Reward model not found at {reward_model_dir}")
        
        # Training parameters
        self.epochs = config["training"]["epochs"]
        self.batch_size = config["training"]["batch_size"]
        self.learning_rate = config["training"].get("learning_rate", 2e-5)
        self.warmup_steps = config["training"].get("warmup_steps", 100)
        self.max_grad_norm = config["training"].get("max_grad_norm", 1.0)
        self.save_steps = config["training"].get("save_steps", 500)
        self.logging_steps = config["training"].get("logging_steps", 50)
        
        # Output directory
        self.output_dir = config["output"]["model_dir"]
        
        # Dataset parameters
        self.max_seq_length = config["dataset"]["max_seq_length"]
        
    def prepare_dataset(self):
        """Prepare ranked dataset for supervised training."""
        self.logger.info("Loading and preparing dataset...")
        
        # Load preference dataset
        dataset_cfg = self.config["dataset"]
        
        train_data = load_dataset(self.tokenizer, dataset_cfg)
        
        # Create ranked dataset
        self.train_dataset = RankedResponseDataset(
            train_data,
            self.reward_model,
            self.tokenizer,
            max_length=self.max_seq_length,
            device=self.device
        )
        
        self.logger.info(f"Prepared {len(self.train_dataset)} high-quality training examples")
    
    def compute_metrics(self, eval_pred: EvalPrediction):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        
        # For causal LM, we compute perplexity
        predictions = predictions.reshape(-1, predictions.shape[-1])
        labels = labels.reshape(-1)
        
        # Mask out padding tokens
        mask = labels != -100
        predictions = predictions[mask]
        labels = labels[mask]
        
        # Compute cross entropy
        loss = F.cross_entropy(torch.tensor(predictions), torch.tensor(labels))
        perplexity = torch.exp(loss)
        
        return {
            "perplexity": perplexity.item(),
            "loss": loss.item()
        }
    
    def train(self):
        """Train the model using supervised learning on reward-ranked responses."""
        try:
            self.logger.info("Starting supervised RLHF training...")
            
            # Prepare dataset
            self.prepare_dataset()
            
            # Training arguments with proper type casting
            training_args = TrainingArguments(
                output_dir=str(self.output_dir),
                num_train_epochs=int(self.epochs),
                per_device_train_batch_size=int(self.batch_size),
                gradient_accumulation_steps=2,
                warmup_steps=int(self.warmup_steps),
                learning_rate=float(self.learning_rate),
                weight_decay=0.01,
                logging_dir=f"{self.output_dir}/logs",
                logging_steps=int(self.logging_steps),
                save_steps=int(self.save_steps),
                save_total_limit=3,
                prediction_loss_only=True,
                remove_unused_columns=False,
                dataloader_pin_memory=False,
                max_grad_norm=float(self.max_grad_norm),
                fp16=False,
                dataloader_num_workers=0,
                load_best_model_at_end=False,  # Disable to avoid evaluation issues
                report_to=None,  # Disable wandb/tensorboard
            )
            
            # Create trainer with minimal configuration
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                processing_class=self.tokenizer,  # Use processing_class instead of tokenizer
            )
            
            # Train
            self.logger.info(f"Training for {self.epochs} epochs on {len(self.train_dataset)} examples")
            trainer.train()
            
            # Save final model
            trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)
            
            # Save training config
            config_path = os.path.join(self.output_dir, "training_config.json")
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            self.logger.info(f"Training completed successfully! Model saved to {self.output_dir}")
            
            # Return training metrics
            return {
                "training_completed": True,
                "final_loss": trainer.state.log_history[-1].get("train_loss", 0),
                "examples_used": len(self.train_dataset),
                "epochs_completed": self.epochs
            }
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise e
