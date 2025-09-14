"""
Reward model trainer with proper pairwise ranking loss for RLHF.
Uses Bradley-Terry model for preference learning.
"""

import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from data.data_loader import load_dataset
from training.reward_model import RewardModel
from utils.logging_utils import setup_logger


class PreferenceRewardTrainer:
    """Trainer for reward model using proper preference data."""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger("preference_reward")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load base model
        model_id = config.get("model") or config.get("base_model")
        self.logger.info(f"Loading base model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.base_model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model = RewardModel(self.base_model).to(self.device)
        
        # Load preference dataset
        ds_cfg = config.get("dataset", {})
        self.logger.info(f"Loading preference dataset: {ds_cfg.get('name')}")
        self.dataset = load_dataset(tokenizer=self.tokenizer, dataset_cfg=ds_cfg)
        
        batch_size = int(config["training"].get("batch_size", 4))
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=self.collate_fn
        )
        
        # Setup optimizer
        lr = float(config["training"].get("learning_rate", 5e-5))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        # Training parameters
        self.epochs = int(config["training"].get("epochs", 3))
        self.logging_steps = int(config["training"].get("logging_steps", 50))
        # Margin for margin-based preference loss (hinge). 0.0 disables margin.
        self.margin = float(config["training"].get("margin", 0.0))
        
    def collate_fn(self, batch):
        """Custom collate function for preference data."""
        chosen_input_ids = torch.stack([item['chosen_input_ids'] for item in batch])
        chosen_attention_mask = torch.stack([item['chosen_attention_mask'] for item in batch])
        rejected_input_ids = torch.stack([item['rejected_input_ids'] for item in batch])
        rejected_attention_mask = torch.stack([item['rejected_attention_mask'] for item in batch])
        
        return {
            'chosen_input_ids': chosen_input_ids,
            'chosen_attention_mask': chosen_attention_mask,
            'rejected_input_ids': rejected_input_ids,
            'rejected_attention_mask': rejected_attention_mask,
            'prompts': [item['prompt'] for item in batch],
            'chosen_texts': [item['chosen_text'] for item in batch],
            'rejected_texts': [item['rejected_text'] for item in batch]
        }
    
    def compute_preference_loss(self, chosen_rewards, rejected_rewards):
        """
        Compute Bradley-Terry preference loss.
        
        Args:
            chosen_rewards: Reward scores for chosen responses [batch_size, 1]
            rejected_rewards: Reward scores for rejected responses [batch_size, 1]
        
        Returns:
            Preference loss
        """
        # Bradley-Terry model: P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
        # With optional margin to enforce a gap between chosen and rejected.
        reward_diff = (chosen_rewards - rejected_rewards).squeeze(-1)  # [batch_size]

        if self.margin > 0.0:
            # Margin ranking (hinge) loss: max(0, margin - (r_c - r_r))
            loss = F.relu(self.margin - reward_diff).mean()
        else:
            # Standard BT loss via BCE on the reward difference logits
            targets = torch.ones_like(reward_diff)
            loss = F.binary_cross_entropy_with_logits(reward_diff, targets)

        return loss
    
    def train_step(self, batch):
        """Single training step."""
        chosen_input_ids = batch['chosen_input_ids'].to(self.device)
        chosen_attention_mask = batch['chosen_attention_mask'].to(self.device)
        rejected_input_ids = batch['rejected_input_ids'].to(self.device)
        rejected_attention_mask = batch['rejected_attention_mask'].to(self.device)
        
        # Get rewards for chosen and rejected responses
        chosen_rewards = self.model(chosen_input_ids, attention_mask=chosen_attention_mask)
        rejected_rewards = self.model(rejected_input_ids, attention_mask=rejected_attention_mask)
        
        # Compute preference loss
        loss = self.compute_preference_loss(chosen_rewards, rejected_rewards)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self):
        """Train the reward model."""
        self.model.train()
        self.logger.info(f"Starting preference reward model training for {self.epochs} epochs")
        
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            correct_preferences = 0
            total_preferences = 0
            
            for step, batch in enumerate(self.dataloader, start=1):
                loss = self.train_step(batch)
                total_loss += loss
                
                # Compute preference accuracy
                with torch.no_grad():
                    chosen_rewards = self.model(
                        batch['chosen_input_ids'].to(self.device),
                        attention_mask=batch['chosen_attention_mask'].to(self.device)
                    )
                    rejected_rewards = self.model(
                        batch['rejected_input_ids'].to(self.device),
                        attention_mask=batch['rejected_attention_mask'].to(self.device)
                    )
                    
                    # Count how often chosen > rejected
                    correct_preferences += (chosen_rewards > rejected_rewards).sum().item()
                    total_preferences += chosen_rewards.size(0)
                
                if step % self.logging_steps == 0:
                    avg_loss = total_loss / step
                    accuracy = correct_preferences / total_preferences if total_preferences > 0 else 0
                    self.logger.info(
                        f"Epoch {epoch} | Step {step} | "
                        f"Loss: {avg_loss:.4f} | Preference Accuracy: {accuracy:.3f}"
                    )
            
            avg_epoch_loss = total_loss / len(self.dataloader)
            final_accuracy = correct_preferences / total_preferences if total_preferences > 0 else 0
            self.logger.info(
                f"Epoch {epoch} completed | "
                f"Avg Loss: {avg_epoch_loss:.4f} | Final Accuracy: {final_accuracy:.3f}"
            )
        
        self._save_model()
        self.logger.info("Preference reward model training complete")
    
    def _save_model(self):
        """Save the trained reward model."""
        output_dir = self.config.get("output", {}).get("model_dir", "models/reward_model_preference")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the base model (needed for AutoModelForCausalLM.from_pretrained)
        self.base_model.save_pretrained(output_dir)
        
        # Save reward model state dict separately
        torch.save(self.model.state_dict(), f"{output_dir}/reward_model.pth")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training config separately
        import json
        with open(f"{output_dir}/training_config.json", "w") as f:
            json.dump(self.config, f, indent=2)
        
        self.logger.info(f"Saved preference reward model to {output_dir}")
    
    def evaluate_preference_accuracy(self, test_dataloader):
        """Evaluate preference accuracy on test data."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_dataloader:
                chosen_rewards = self.model(
                    batch['chosen_input_ids'].to(self.device),
                    attention_mask=batch['chosen_attention_mask'].to(self.device)
                )
                rejected_rewards = self.model(
                    batch['rejected_input_ids'].to(self.device),
                    attention_mask=batch['rejected_attention_mask'].to(self.device)
                )
                
                correct += (chosen_rewards > rejected_rewards).sum().item()
                total += chosen_rewards.size(0)
        
        accuracy = correct / total if total > 0 else 0
        self.logger.info(f"Preference accuracy: {accuracy:.3f} ({correct}/{total})")
        return accuracy
