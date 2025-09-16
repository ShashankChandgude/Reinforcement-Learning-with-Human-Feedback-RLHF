"""
Balanced reward model trainer - Best of both worlds approach.
Combines original model's learning quality with optimized model's GPU efficiency.
"""

import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from data.data_loader import load_dataset
from training.reward_model import RewardModel
from utils.logging_utils import setup_logger
import time


class BalancedRewardTrainer:
    """Balanced trainer combining accuracy and efficiency."""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger("balanced_reward")
        
        # GPU setup with optimizations
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            # Optimize GPU settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Clear any existing cache
            torch.cuda.empty_cache()
        else:
            self.device = torch.device("cpu")
            self.logger.warning("GPU not available, using CPU")
        
        # Load model (similar to original for accuracy)
        model_id = config.get("model") or config.get("base_model")
        self.logger.info(f"Loading model: {model_id}")
        
        # Use fast tokenizer for speed
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            use_fast=config.get("optimization", {}).get("use_fast_tokenizer", True)
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model with fp32 for stability
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # GTX 1650 works better with fp32
        )
        
        self.model = RewardModel(self.base_model).to(self.device)
        
        # Conditional optimizations based on config
        opt_config = config.get("optimization", {})
        
        # Only enable gradient checkpointing if explicitly requested
        if opt_config.get("gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()
            self.logger.info("Gradient checkpointing enabled")
        
        # Load dataset
        ds_cfg = config.get("dataset", {})
        self.logger.info(f"Loading dataset: {ds_cfg.get('name')} (size: {ds_cfg.get('subset_size')})")
        self.dataset = load_dataset(tokenizer=self.tokenizer, dataset_cfg=ds_cfg)
        
        # Setup dataloader with optimizations
        training_cfg = config.get("training", {})
        batch_size = int(training_cfg.get("batch_size", 2))
        num_workers = int(training_cfg.get("dataloader_num_workers", 2))
        pin_memory = training_cfg.get("pin_memory", True)
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn
        )
        
        # Training parameters
        self.epochs = int(training_cfg.get("epochs", 5))
        self.gradient_accumulation_steps = int(training_cfg.get("gradient_accumulation_steps", 4))
        self.logging_steps = int(training_cfg.get("logging_steps", 15))
        self.margin = float(training_cfg.get("margin", 0.5))
        
        # Setup optimizer (same as original for consistency)
        lr = float(training_cfg.get("learning_rate", 3e-5))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        # Mixed precision setup (disabled for GTX 1650)
        self.use_mixed_precision = training_cfg.get("mixed_precision", False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_mixed_precision else None
        
        self.logger.info(f"Trainer initialized:")
        self.logger.info(f"  Dataset size: {len(self.dataset)}")
        self.logger.info(f"  Batch size: {batch_size}")
        self.logger.info(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        self.logger.info(f"  Effective batch size: {batch_size * self.gradient_accumulation_steps}")
        self.logger.info(f"  Learning rate: {lr}")
        self.logger.info(f"  Mixed precision: {self.use_mixed_precision}")
    
    def collate_fn(self, batch):
        """Collate function for preference pairs."""
        chosen_texts = [item['chosen_text'] for item in batch]
        rejected_texts = [item['rejected_text'] for item in batch]
        
        # Tokenize with proper settings
        chosen_encoded = self.tokenizer(
            chosen_texts,
            padding=True,
            truncation=True,
            max_length=self.config['dataset']['max_seq_length'],
            return_tensors='pt'
        )
        
        rejected_encoded = self.tokenizer(
            rejected_texts,
            padding=True,
            truncation=True,
            max_length=self.config['dataset']['max_seq_length'],
            return_tensors='pt'
        )
        
        return {
            'chosen_input_ids': chosen_encoded['input_ids'],
            'chosen_attention_mask': chosen_encoded['attention_mask'],
            'rejected_input_ids': rejected_encoded['input_ids'],
            'rejected_attention_mask': rejected_encoded['attention_mask']
        }
    
    def compute_preference_loss(self, chosen_rewards, rejected_rewards):
        """Compute Bradley-Terry preference loss."""
        # Ensure rewards are scalars
        if chosen_rewards.dim() > 1:
            chosen_rewards = chosen_rewards.squeeze()
        if rejected_rewards.dim() > 1:
            rejected_rewards = rejected_rewards.squeeze()
        
        # Bradley-Terry loss with margin
        logits = chosen_rewards - rejected_rewards - self.margin
        loss = -F.logsigmoid(logits).mean()
        return loss
    
    def train_step(self, batch, step):
        """Single training step - similar to original for accuracy."""
        # Move to device with non_blocking for speed
        chosen_input_ids = batch['chosen_input_ids'].to(self.device, non_blocking=True)
        chosen_attention_mask = batch['chosen_attention_mask'].to(self.device, non_blocking=True)
        rejected_input_ids = batch['rejected_input_ids'].to(self.device, non_blocking=True)
        rejected_attention_mask = batch['rejected_attention_mask'].to(self.device, non_blocking=True)
        
        # Forward pass (with optional mixed precision)
        if self.use_mixed_precision and self.scaler is not None:
            with torch.cuda.amp.autocast():
                chosen_rewards = self.model(chosen_input_ids, attention_mask=chosen_attention_mask)
                rejected_rewards = self.model(rejected_input_ids, attention_mask=rejected_attention_mask)
                loss = self.compute_preference_loss(chosen_rewards, rejected_rewards)
                loss = loss / self.gradient_accumulation_steps
        else:
            chosen_rewards = self.model(chosen_input_ids, attention_mask=chosen_attention_mask)
            rejected_rewards = self.model(rejected_input_ids, attention_mask=rejected_attention_mask)
            loss = self.compute_preference_loss(chosen_rewards, rejected_rewards)
            loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        if self.use_mixed_precision and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimizer step with gradient accumulation
        if (step + 1) % self.gradient_accumulation_steps == 0:
            if self.use_mixed_precision and self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        return loss.item() * self.gradient_accumulation_steps  # Return unscaled loss
    
    def compute_accuracy(self, batch):
        """Compute preference accuracy."""
        with torch.no_grad():
            chosen_input_ids = batch['chosen_input_ids'].to(self.device)
            chosen_attention_mask = batch['chosen_attention_mask'].to(self.device)
            rejected_input_ids = batch['rejected_input_ids'].to(self.device)
            rejected_attention_mask = batch['rejected_attention_mask'].to(self.device)
            
            chosen_rewards = self.model(chosen_input_ids, attention_mask=chosen_attention_mask)
            rejected_rewards = self.model(rejected_input_ids, attention_mask=rejected_attention_mask)
            
            # Count correct preferences
            correct = (chosen_rewards > rejected_rewards).sum().item()
            total = chosen_rewards.size(0)
            
            return correct, total
    
    def train(self):
        """Training loop - balanced approach."""
        self.model.train()
        start_time = time.time()
        self.logger.info(f"Starting balanced reward model training for {self.epochs} epochs")
        
        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()
            total_loss = 0.0
            correct_preferences = 0
            total_preferences = 0
            
            for step, batch in enumerate(self.dataloader):
                loss = self.train_step(batch, step)
                total_loss += loss
                
                # Compute accuracy at logging intervals (like original)
                if step % self.logging_steps == 0:
                    correct, total = self.compute_accuracy(batch)
                    correct_preferences += correct
                    total_preferences += total
                    
                    current_accuracy = correct / total if total > 0 else 0.0
                    elapsed = time.time() - epoch_start
                    
                    self.logger.info(
                        f"Epoch {epoch} | Step {step} | "
                        f"Loss: {loss:.4f} | "
                        f"Preference Accuracy: {current_accuracy:.3f} | "
                        f"Time: {elapsed:.1f}s"
                    )
            
            # Epoch summary
            avg_loss = total_loss / len(self.dataloader)
            epoch_accuracy = correct_preferences / total_preferences if total_preferences > 0 else 0.0
            epoch_time = time.time() - epoch_start
            
            self.logger.info(
                f"Epoch {epoch} completed in {epoch_time:.1f}s | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"Final Accuracy: {epoch_accuracy:.3f}"
            )
            
            # Clear cache between epochs if configured
            if self.config.get("optimization", {}).get("clear_cache_between_epochs", False):
                torch.cuda.empty_cache()
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.1f}s")
        
        # Save model
        self.save_model()
    
    def save_model(self):
        """Save the trained model."""
        output_dir = self.config["output"]["model_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the reward model weights
        reward_model_path = os.path.join(output_dir, "reward_model.pth")
        torch.save(self.model.state_dict(), reward_model_path)
        
        # Save the base model and tokenizer
        self.base_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training config
        import json
        config_path = os.path.join(output_dir, "training_config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)
        
        self.logger.info(f"Model saved to {output_dir}")
    
    def cleanup_memory(self):
        """Clean up memory."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'base_model'):
            del self.base_model
        torch.cuda.empty_cache()
        self.logger.info("Memory cleaned up")


def run_balanced_reward_training(config_path: str):
    """Run balanced reward model training."""
    from utils.config_loader import load_config
    
    config = load_config(config_path)
    trainer = BalancedRewardTrainer(config)
    
    try:
        trainer.train()
        return True
    except Exception as e:
        trainer.logger.error(f"Training failed: {e}")
        return False
    finally:
        trainer.cleanup_memory()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "configs/reward_preference_balanced.yaml"
    
    success = run_balanced_reward_training(config_path)
    exit(0 if success else 1)
