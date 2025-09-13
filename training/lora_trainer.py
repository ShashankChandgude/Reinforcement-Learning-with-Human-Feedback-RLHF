"""
LoRA fine-tuning trainer for larger models with memory efficiency.
"""

import os
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from data.data_loader import load_dataset
from utils.logging_utils import setup_logger
from typing import Dict, Any


class LoRATrainer:
    """LoRA fine-tuning trainer for memory-efficient training on larger models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logger("lora_training")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        model_name = config["model"]
        self.logger.info(f"Loading model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Setup LoRA configuration
        if config.get("use_lora", True):
            lora_config = LoraConfig(
                r=config["lora_config"]["r"],
                lora_alpha=config["lora_config"]["lora_alpha"],
                target_modules=config["lora_config"]["target_modules"],
                lora_dropout=config["lora_config"]["lora_dropout"],
                bias=config["lora_config"]["bias"],
                task_type=TaskType.CAUSAL_LM,
            )
            
            self.model = get_peft_model(self.base_model, lora_config)
            self.logger.info("LoRA configuration applied")
        else:
            self.model = self.base_model
        
        # Load dataset
        ds_cfg = config.get("dataset", {})
        self.logger.info(f"Loading dataset: {ds_cfg.get('name')}")
        self.dataset = load_dataset(tokenizer=self.tokenizer, dataset_cfg=ds_cfg)
        
        # Setup training arguments
        self.training_args = TrainingArguments(
            output_dir=config["output"]["model_dir"],
            logging_dir=config["output"]["log_dir"],
            num_train_epochs=config["training"]["epochs"],
            per_device_train_batch_size=config["training"]["batch_size"],
            per_device_eval_batch_size=config["training"]["batch_size"],
            learning_rate=config["training"]["learning_rate"],
            logging_steps=config["training"]["logging_steps"],
            save_steps=config["training"]["save_steps"],
            eval_steps=config["training"]["eval_steps"],
            gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
            max_grad_norm=config["training"]["max_grad_norm"],
            warmup_steps=config["training"]["warmup_steps"],
            save_total_limit=2,
            load_best_model_at_end=True,
            eval_strategy="steps",  # Changed from evaluation_strategy
            save_strategy="steps",
            logging_strategy="steps",
            remove_unused_columns=False,
            report_to=None,  # Disable W&B and other reporting
        )
        
        # Setup data collator
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
    
    def prepare_dataset_for_training(self):
        """Prepare dataset for causal language modeling."""
        # Check if it's a PreferenceDataset (PyTorch Dataset) or HuggingFace Dataset
        if hasattr(self.dataset, 'prompts'):
            # It's a PreferenceDataset - convert to HuggingFace format
            from datasets import Dataset
            
            # Prepare texts from preference data
            texts = []
            for i in range(len(self.dataset)):
                prompt = self.dataset.prompts[i]
                chosen = self.dataset.chosen_responses[i]
                texts.append(f"{prompt} {chosen}")
            
            # Create HuggingFace dataset
            hf_dataset = Dataset.from_dict({"text": texts})
            
            def tokenize_function(examples):
                tokenized = self.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=True,
                    max_length=self.config["dataset"]["max_seq_length"],
                    return_tensors=None  # Don't return tensors for HuggingFace datasets
                )
                # Handle batched tokenization - each value is a list of lists
                for key in tokenized:
                    if isinstance(tokenized[key], list) and len(tokenized[key]) > 0:
                        if isinstance(tokenized[key][0], list):
                            # It's a list of lists (batched), convert each sublist
                            tokenized[key] = [[int(x) for x in sublist] for sublist in tokenized[key]]
                        else:
                            # It's a single list, convert directly
                            tokenized[key] = [int(x) for x in tokenized[key]]
                return tokenized
            
            # Tokenize dataset
            tokenized_dataset = hf_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=hf_dataset.column_names
            )
        else:
            # It's a HuggingFace dataset
            def tokenize_function(examples):
                # For preference data, we'll use the chosen responses for training
                if 'chosen_text' in examples:
                    texts = [f"{p} {r}" for p, r in zip(examples['prompt'], examples['chosen_text'])]
                else:
                    # Fallback for other dataset types
                    texts = examples.get('text', examples.get('input_ids', []))
                
                return self.tokenizer(
                    texts,
                    truncation=True,
                    padding=True,
                    max_length=self.config["dataset"]["max_seq_length"],
                    return_tensors=None  # Don't return tensors for HuggingFace datasets
                )
            
            # Tokenize dataset
            tokenized_dataset = self.dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=self.dataset.column_names
            )
        
        # Split into train/eval
        train_size = int(0.8 * len(tokenized_dataset))
        eval_size = len(tokenized_dataset) - train_size
        
        train_dataset = tokenized_dataset.select(range(train_size))
        eval_dataset = tokenized_dataset.select(range(train_size, train_size + eval_size))
        
        return train_dataset, eval_dataset
    
    def train(self):
        """Train the model with LoRA."""
        self.logger.info("Preparing dataset for training...")
        train_dataset, eval_dataset = self.prepare_dataset_for_training()
        
        self.logger.info(f"Training dataset size: {len(train_dataset)}")
        self.logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator,
        )
        
        # Print model info
        self.logger.info("Model parameters:")
        if hasattr(self.model, 'print_trainable_parameters'):
            self.model.print_trainable_parameters()
        
        # Start training
        self.logger.info("Starting LoRA training...")
        trainer.train()
        
        # Save model
        self.logger.info("Saving model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config["output"]["model_dir"])
        
        # Save LoRA adapters separately
        if self.config.get("use_lora", True):
            self.model.save_pretrained(f"{self.config['output']['model_dir']}/lora_adapters")
            self.logger.info("LoRA adapters saved separately")
        
        self.logger.info("LoRA training completed!")
        
        return trainer


def run_lora_training(config_path: str):
    """Run LoRA training with given configuration."""
    from utils.config_loader import load_config
    
    config = load_config(config_path)
    trainer = LoRATrainer(config)
    return trainer.train()


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/lora_7b_config.yaml"
    run_lora_training(config_path)
