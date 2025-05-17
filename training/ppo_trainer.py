import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from RewardModel import RewardModel
from DataLoader import load_dolly_dataset
from utils.logging_utils import setup_logger

class PPOTrainer:
    """
    Trainer for Proximal Policy Optimization (PPO) fine-tuning.
    """
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger("ppo")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model().to(self.device)
        self.reward_model = self._load_reward_model().to(self.device)
        self.dataloader = self._prepare_dataloader()
        self.optimizer = self._configure_optimizer()
        training_cfg = config.get("training", {})
        self.epochs = training_cfg.get("epochs", 3)
        self.clip_epsilon = training_cfg.get("clip_epsilon", 0.2)
        self.logging_steps = training_cfg.get("logging_steps", 50)

    def _load_tokenizer(self):
        model_id = self.config.get("model")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_model(self):
        model_id = self.config.get("model")
        return AutoModelForCausalLM.from_pretrained(model_id)

    def _load_reward_model(self):
        rm_id = self.config.get("reward_model_dir")
        base = AutoModelForCausalLM.from_pretrained(rm_id)
        return RewardModel(base)

    def _prepare_dataloader(self):
        dc = self.config.get("dataset", {})
        data = load_dolly_dataset(
            tokenizer=self.tokenizer,
            subset_size=dc.get("subset_size", 5000),
            max_length=dc.get("max_seq_length", 512)
        )
        bs = self.config.get("training", {}).get("batch_size", 4)
        return DataLoader(data, batch_size=bs, shuffle=True)

    def _configure_optimizer(self):
        lr = self.config.get("training", {}).get("learning_rate", 5e-5)
        return torch.optim.AdamW(self.model.parameters(), lr=lr)

    def train(self):
        self.model.train()
        self.reward_model.eval()
        self.logger.info(f"Starting PPO training for {self.epochs} epochs")
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            for step, batch in enumerate(self.dataloader, start=1):
                loss = self._train_step(batch)
                total_loss += loss
                if step % self.logging_steps == 0:
                    avg_loss = total_loss / step
                    self.logger.info(f"Epoch {epoch} | Step {step} | Avg PPO Loss: {avg_loss:.4f}")
            avg_epoch_loss = total_loss / len(self.dataloader)
            self.logger.info(f"Epoch {epoch} completed | Avg PPO Loss: {avg_epoch_loss:.4f}")
        self._save_model()

    def _train_step(self, batch):
        input_ids, attention_mask = batch
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        actions = torch.argmax(probs, dim=-1)
        log_probs = torch.log(torch.gather(probs, -1, actions.unsqueeze(-1)).squeeze(-1) + 1e-10)
        rewards = self.reward_model(input_ids, attention_mask=attention_mask).detach()
        advantage = rewards - rewards.mean()
        ratio = torch.exp(log_probs - log_probs.detach())
        clipped = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage
        ppo_loss = -torch.min(ratio * advantage, clipped).mean()
        self.optimizer.zero_grad()
        ppo_loss.backward()
        self.optimizer.step()
        return ppo_loss.item()

    def _save_model(self):
        out_dir = self.config.get("output", {}).get("model_dir", "models/ppo_model")
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        self.logger.info(f"Saved PPO model to {out_dir}")
