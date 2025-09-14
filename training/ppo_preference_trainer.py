"""
PPO trainer with proper preference-based reward model integration.
"""

import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
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
        # Simple value function head on top of policy hidden states (last token)
        self.value_head = nn.Linear(self.model.config.hidden_size, 1).to(self.device)
        
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
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.value_head.parameters()), lr=lr
        )
        
        # Training parameters with explicit type conversion
        self.epochs = int(config["training"]["epochs"])
        self.clip_epsilon = float(config["training"]["clip_epsilon"])
        self.logging_steps = int(config["training"]["logging_steps"])
        self.value_coeff = float(config["training"].get("value_coeff", 0.5))
        self.entropy_coeff = float(config["training"].get("entropy_coeff", 0.01))
        self.gamma = float(config["training"].get("gamma", 0.99))
        self.gae_lambda = float(config["training"].get("gae_lambda", 0.95))
        self.save_steps = int(config["training"].get("save_steps", 100))
        
        # Tracking
        self.training_history = {
            "ppo_losses": [],
            "value_losses": [],
            "total_losses": [],
            "rewards": [],
            "values": [],
            "advantages": [],
            "kl_divergences": [],
            "entropy": [],
            "grad_norm": [],
            "preference_accuracies": []
        }
        
        # PPO-specific tracking
        self.old_log_probs_buffer = {}  # Store old log probs for PPO
    
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

    def compute_values(self, input_ids, attention_mask):
        """Compute state values from policy hidden states (last token)."""
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        values = self.value_head(last_hidden)
        return values  # [batch, 1]

    def compute_gae(self, rewards, values, next_values=None):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        For single-step episodes (i.i.d. samples), we use:
        - delta = reward - value (TD error)
        - advantage = delta (no trajectory to bootstrap)
        - return = reward (target for value function)
        
        Args:
            rewards: [batch_size] - immediate rewards
            values: [batch_size] - current state values
            next_values: [batch_size] - next state values (None for terminal states)
        """
        rewards = rewards.detach()
        values = values.detach().squeeze(-1) if values.dim() > 1 else values.detach()
        
        # For i.i.d. samples, we treat each as a terminal state
        # TD error: δ = r + γ * V(s') - V(s), but V(s') = 0 for terminal states
        td_errors = rewards - values
        
        # For GAE with terminal states: A = δ (no future states to consider)
        advantages = td_errors
        
        # Returns are the targets for value function training
        returns = rewards  # For terminal states, return = reward
        
        return advantages, returns
    
    def compute_kl_divergence(self, old_log_probs, new_log_probs):
        """Compute KL divergence between old and new policies."""
        return (old_log_probs - new_log_probs).mean()
    
    def ppo_loss(self, log_probs, old_log_probs, advantages, entropy=None):
        """
        Compute PPO loss with clipped surrogate objective and optional entropy regularization.
        
        Args:
            log_probs: [batch_size] - current policy log probabilities
            old_log_probs: [batch_size] - old policy log probabilities  
            advantages: [batch_size] - advantage estimates
            entropy: [batch_size] - entropy of current policy (optional)
        """
        # Ensure all tensors are properly shaped and detached where needed
        log_probs = log_probs.squeeze() if log_probs.dim() > 1 else log_probs
        old_log_probs = old_log_probs.squeeze() if old_log_probs.dim() > 1 else old_log_probs
        advantages = advantages.detach()  # Don't backprop through advantages
        
        # Compute importance sampling ratio
        log_ratio = log_probs - old_log_probs
        ratio = torch.exp(torch.clamp(log_ratio, -10, 10))  # Prevent extreme ratios
        
        # PPO clipped surrogate objective
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        
        # Compute surrogate losses
        surrogate1 = ratio * advantages
        surrogate2 = clipped_ratio * advantages
        
        # Take minimum (most conservative estimate)
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        
        # Add entropy regularization if provided
        entropy_coeff = getattr(self, 'entropy_coeff', 0.01)
        if entropy is not None and entropy_coeff > 0:
            entropy_loss = -entropy_coeff * entropy.mean()
            policy_loss = policy_loss + entropy_loss
        
        # Robust handling of numerical issues
        if torch.isnan(policy_loss) or torch.isinf(policy_loss):
            self.logger.warning("NaN/Inf detected in policy loss, using fallback")
            policy_loss = torch.tensor(0.01, requires_grad=True, device=policy_loss.device)
        
        return policy_loss
    
    def compute_action_log_probs_and_entropy(self, logits, actions):
        """
        Compute action log probabilities and entropy from logits.
        
        Args:
            logits: [batch, seq_len, vocab_size] - model logits
            actions: [batch, seq_len] - selected actions (token ids)
        
        Returns:
            log_probs: [batch] - sum of log probabilities for the sequence
            entropy: [batch] - entropy of the policy distribution
        """
        # Convert logits to log probabilities
        log_probs_dist = F.log_softmax(logits, dim=-1)  # [batch, seq_len, vocab_size]
        probs_dist = F.softmax(logits, dim=-1)  # [batch, seq_len, vocab_size]
        
        # Get log probabilities for actual actions
        action_log_probs = torch.gather(log_probs_dist, -1, actions.unsqueeze(-1)).squeeze(-1)  # [batch, seq_len]
        
        # Sum over sequence length to get total log probability
        sequence_log_probs = action_log_probs.sum(dim=-1)  # [batch]
        
        # Compute entropy: -sum(p * log(p))
        entropy = -(probs_dist * log_probs_dist).sum(dim=-1).mean(dim=-1)  # [batch]
        
        return sequence_log_probs, entropy
    
    def train_step(self, batch):
        """Single PPO training step with improved computations."""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Get current policy outputs
        outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        logits = outputs.logits
        
        # Compute action log probabilities and entropy
        action_log_probs, entropy = self.compute_action_log_probs_and_entropy(logits, input_ids)
        
        # Compute rewards and values
        rewards = self.compute_reward(input_ids, attention_mask).squeeze(-1)
        values = self.compute_values(input_ids, attention_mask).squeeze(-1)

        # Compute advantages and returns using GAE
        advantages, returns = self.compute_gae(rewards, values)
        
        # Normalize advantages for stability (important for PPO)
        if len(advantages) > 1:
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            if adv_std > 1e-8:  # Avoid division by zero
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)
            else:
                # If std is too small, just center the advantages
                advantages = advantages - adv_mean
        
        # Handle old log probs for PPO (better tracking mechanism)
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        batch_key = f"{batch_size}x{seq_len}"
        
        if batch_key in self.old_log_probs_buffer:
            old_log_probs = self.old_log_probs_buffer[batch_key]
            # Ensure shapes match
            if old_log_probs.shape != action_log_probs.shape:
                old_log_probs = action_log_probs.detach() + torch.randn_like(action_log_probs) * 0.01
        else:
            # For first iteration, use slightly perturbed current log probs
            old_log_probs = action_log_probs.detach() + torch.randn_like(action_log_probs) * 0.01
        
        # Store current log probs for next iteration
        self.old_log_probs_buffer[batch_key] = action_log_probs.detach().clone()
        
        # Compute PPO policy loss with entropy
        policy_loss = self.ppo_loss(action_log_probs, old_log_probs, advantages, entropy)
        
        # Compute value loss (MSE between predicted values and returns)
        value_loss = F.mse_loss(values, returns.detach())
        
        # Total loss with value function coefficient
        total_loss = policy_loss + self.value_coeff * value_loss
        
        # Compute KL divergence for monitoring
        kl_div = self.compute_kl_divergence(old_log_probs, action_log_probs)
        
        # Backward pass with gradient clipping
        self.optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.value_head.parameters()), 
            max_norm=1.0
        )
        self.optimizer.step()
        
        # Compute preference accuracy for monitoring
        preference_accuracy = 0.0
        if 'chosen_texts' in batch and 'rejected_texts' in batch:
            # For preference data, we can compute how well rewards align with preferences
            with torch.no_grad():
                # This is a simplified metric - in practice you'd want to compare
                # chosen vs rejected responses properly
                preference_accuracy = 0.5  # Placeholder
        
        return {
            'ppo_losses': policy_loss.item(),
            'value_losses': value_loss.item(),
            'total_losses': total_loss.item(),
            'rewards': rewards.mean().item(),
            'values': values.mean().item(),
            'advantages': advantages.mean().item(),
            'kl_divergences': kl_div.item(),
            'entropy': entropy.mean().item(),
            'grad_norm': grad_norm.item(),
            'preference_accuracies': preference_accuracy
        }
    
    def train(self):
        """Train the model with PPO."""
        self.model.train()
        self.logger.info(f"Starting PPO training for {self.epochs} epochs")
        
        global_step = 0
        
        for epoch in range(1, self.epochs + 1):
            epoch_metrics = {key: [] for key in self.training_history.keys()}
            
            for step, batch in enumerate(self.dataloader, start=1):
                metrics = self.train_step(batch)
                global_step += 1
                
                # Store metrics
                for key, value in metrics.items():
                    if key in epoch_metrics:
                        epoch_metrics[key].append(value)
                        self.training_history[key].append(value)
                
                # Logging
                if step % self.logging_steps == 0:
                    avg_metrics = {k: sum(v) / len(v) for k, v in epoch_metrics.items() if v}
                    log_message = (
                        f"Epoch {epoch} | Step {step} | "
                        f"PPO Loss: {avg_metrics.get('ppo_losses', 0):.4f} | "
                        f"Value Loss: {avg_metrics.get('value_losses', 0):.4f} | "
                        f"Total Loss: {avg_metrics.get('total_losses', 0):.4f} | "
                        f"Reward: {avg_metrics.get('rewards', 0):.4f} | "
                        f"Value: {avg_metrics.get('values', 0):.4f} | "
                        f"Advantage: {avg_metrics.get('advantages', 0):.4f} | "
                        f"KL Div: {avg_metrics.get('kl_divergences', 0):.4f} | "
                        f"Entropy: {avg_metrics.get('entropy', 0):.4f} | "
                        f"Grad Norm: {avg_metrics.get('grad_norm', 0):.4f}"
                    )
                    self.logger.info(log_message)
                
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
