"""
PPO trainer with proper preference-based reward model integration.
"""

import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from data.data_loader import load_dataset
from training.reward_model import RewardModel
from utils.logging_utils import setup_logger
from typing import Dict, Any
import json


class PPOPreferenceTrainer:
    """PPO trainer that uses preference-based reward model."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logger("ppo_preference")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load base model
        model_name = config["model"]
        self.logger.info(f"Loading base model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
        # Load reward model
        reward_model_dir = config.get("reward_model_dir")
        if reward_model_dir and os.path.exists(reward_model_dir):
            self.logger.info(f"Loading reward model from: {reward_model_dir}")
            base_rm = AutoModelForCausalLM.from_pretrained(reward_model_dir)
            self.reward_model = RewardModel(base_rm).to(self.device)
            
            # Load reward model weights
            reward_weights_path = f"{reward_model_dir}/reward_model.pth"
            if os.path.exists(reward_weights_path):
                reward_state = torch.load(reward_weights_path, map_location=self.device)
                self.reward_model.load_state_dict(reward_state)
                self.logger.info("Reward model weights loaded")
            else:
                self.logger.warning("Reward model weights not found, using untrained model")
        else:
            self.logger.warning("No reward model directory specified, using untrained model")
            base_rm = AutoModelForCausalLM.from_pretrained(model_name)
            self.reward_model = RewardModel(base_rm).to(self.device)
        
        self.reward_model.eval()
        
        # Load dataset
        ds_cfg = config.get("dataset", {})
        self.logger.info(f"Loading dataset: {ds_cfg.get('name')}")
        self.dataset = load_dataset(tokenizer=self.tokenizer, dataset_cfg=ds_cfg)
        
        # Setup DataLoader with explicit type conversion
        batch_size = int(config["training"]["batch_size"])
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=self.collate_fn
        )
        
        # Setup optimizer with explicit type conversion
        lr = float(config["training"]["learning_rate"])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        # Training parameters with explicit type conversion
        self.epochs = int(config["training"]["epochs"])
        self.clip_epsilon = float(config["training"]["clip_epsilon"])
        self.logging_steps = int(config["training"]["logging_steps"])
        self.save_steps = int(config["training"].get("save_steps", 100))
        
        # Tracking
        self.training_history = {
            "ppo_losses": [],
            "rewards": [],
            "kl_divergences": [],
            "preference_accuracies": []
        }
    
    def collate_fn(self, batch):
        """Custom collate function for preference data."""
        if 'chosen_input_ids' in batch[0]:
            # Preference data
            chosen_input_ids = torch.stack([item['chosen_input_ids'] for item in batch])
            chosen_attention_mask = torch.stack([item['chosen_attention_mask'] for item in batch])
            return {
                'input_ids': chosen_input_ids,
                'attention_mask': chosen_attention_mask,
                'prompts': [item['prompt'] for item in batch],
                'chosen_texts': [item['chosen_text'] for item in batch],
                'rejected_texts': [item['rejected_text'] for item in batch]
            }
        else:
            # Regular data
            input_ids = torch.stack([item['input_ids'] for item in batch])
            attention_mask = torch.stack([item['attention_mask'] for item in batch])
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
    
    def compute_reward(self, input_ids, attention_mask):
        """Compute reward using the reward model."""
        with torch.no_grad():
            rewards = self.reward_model(input_ids, attention_mask)
        return rewards
    
    def compute_kl_divergence(self, old_log_probs, new_log_probs):
        """Compute KL divergence between old and new policies."""
        return (old_log_probs - new_log_probs).mean()
    
    def ppo_loss(self, log_probs, old_log_probs, rewards, advantages):
        """Compute PPO loss."""
        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        return policy_loss
    
    def train_step(self, batch):
        """Single PPO training step."""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Get current policy outputs
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
        logits = outputs.logits
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = torch.gather(log_probs, -1, input_ids.unsqueeze(-1)).squeeze(-1)
        
        # Compute rewards
        rewards = self.compute_reward(input_ids, attention_mask).squeeze(-1)
        
        # Compute advantages (simplified - in practice you'd use GAE)
        advantages = rewards - rewards.mean()
        
        # For the first step, use current log probs as "old" log probs
        # In practice, you'd store old log probs from previous iterations
        old_log_probs = action_log_probs.detach()
        
        # Compute PPO loss
        ppo_loss = self.ppo_loss(action_log_probs, old_log_probs, rewards, advantages)
        
        # Compute KL divergence
        kl_div = self.compute_kl_divergence(old_log_probs, action_log_probs)
        
        # Backward pass
        self.optimizer.zero_grad()
        ppo_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Compute preference accuracy if we have preference data
        preference_accuracy = 0.0
        if 'chosen_texts' in batch and 'rejected_texts' in batch:
            # Generate responses for chosen and rejected prompts
            chosen_rewards = self.compute_reward(input_ids, attention_mask)
            # For rejected, we'd need to generate them, but for simplicity, use current rewards
            rejected_rewards = chosen_rewards  # Placeholder
            
            correct_preferences = (chosen_rewards > rejected_rewards).sum().item()
            total_preferences = chosen_rewards.size(0)
            preference_accuracy = correct_preferences / total_preferences if total_preferences > 0 else 0.0
        
        return {
            'ppo_losses': ppo_loss.item(),
            'rewards': rewards.mean().item(),
            'kl_divergences': kl_div.item(),
            'preference_accuracies': preference_accuracy
        }
    
    def train(self):
        """Train the model with PPO."""
        self.model.train()
        self.logger.info(f"Starting PPO training for {self.epochs} epochs")
        
        global_step = 0
        
        for epoch in range(1, self.epochs + 1):
            epoch_metrics = {
                'ppo_losses': [],
                'rewards': [],
                'kl_divergences': [],
                'preference_accuracies': []
            }
            
            for step, batch in enumerate(self.dataloader, start=1):
                metrics = self.train_step(batch)
                global_step += 1
                
                # Store metrics
                for key, value in metrics.items():
                    epoch_metrics[key].append(value)
                    self.training_history[key].append(value)
                
                # Logging
                if step % self.logging_steps == 0:
                    avg_metrics = {k: sum(v) / len(v) for k, v in epoch_metrics.items()}
                    self.logger.info(
                        f"Epoch {epoch} | Step {step} | "
                        f"PPO Loss: {avg_metrics['ppo_losses']:.4f} | "
                        f"Reward: {avg_metrics['rewards']:.4f} | "
                        f"KL Div: {avg_metrics['kl_divergences']:.4f} | "
                        f"Pref Acc: {avg_metrics['preference_accuracies']:.4f}"
                    )
                
                # Save checkpoint
                if global_step % self.save_steps == 0:
                    self.save_checkpoint(global_step)
            
            # Epoch summary
            avg_epoch_metrics = {k: sum(v) / len(v) for k, v in epoch_metrics.items()}
            self.logger.info(f"Epoch {epoch} completed:")
            for metric, value in avg_epoch_metrics.items():
                self.logger.info(f"  {metric}: {value:.4f}")
        
        # Save final model
        self.save_model()
        self.save_training_history()
        
        self.logger.info("PPO training completed!")
    
    def save_checkpoint(self, step):
        """Save model checkpoint."""
        checkpoint_dir = f"{self.config['output']['model_dir']}/checkpoint-{step}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        self.logger.info(f"Checkpoint saved at step {step}")
    
    def save_model(self):
        """Save final model."""
        output_dir = self.config["output"]["model_dir"]
        os.makedirs(output_dir, exist_ok=True)
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save config
        with open(f"{output_dir}/config.json", "w") as f:
            json.dump(self.config, f, indent=2)
        
        self.logger.info(f"Final model saved to {output_dir}")
    
    def save_training_history(self):
        """Save training history."""
        history_file = f"{self.config['output']['model_dir']}/training_history.json"
        with open(history_file, "w") as f:
            json.dump(self.training_history, f, indent=2)
        
        self.logger.info(f"Training history saved to {history_file}")


def run_ppo_training(config_path: str):
    """Run PPO training with given configuration."""
    from utils.config_loader import load_config
    
    config = load_config(config_path)
    trainer = PPOPreferenceTrainer(config)
    return trainer.train()


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/ppo_preference.yaml"
    run_ppo_training(config_path)
