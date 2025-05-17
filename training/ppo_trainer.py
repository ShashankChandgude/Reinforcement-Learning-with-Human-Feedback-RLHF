import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from data.data_loader import load_dataset
from RewardModel import RewardModel
from utils.logging_utils import setup_logger

class PPOTrainer:
    """Trainer for Proximal Policy Optimization fine-tuning."""
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger("ppo")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_id = config["model"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device)

        reward_dir = config.get("reward_model_dir")
        base_rm = AutoModelForCausalLM.from_pretrained(reward_dir)
        self.reward_model = RewardModel(base_rm).to(self.device)

        ds_cfg = config.get("dataset", {})
        self.logger.info(f"Loading dataset with loader: {ds_cfg.get('loader')} name: {ds_cfg.get('name')}")
        dataset = load_dataset(tokenizer=self.tokenizer, dataset_cfg=ds_cfg)
        bs = config.get("training", {}).get("batch_size", 4)
        self.dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)

        lr = config.get("training", {}).get("learning_rate", 5e-5)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        tcfg = config.get("training", {})
        self.epochs = tcfg.get("epochs", 3)
        self.clip_epsilon = tcfg.get("clip_epsilon", 0.2)
        self.logging_steps = tcfg.get("logging_steps", 50)

    def train(self):
        self.model.train()
        self.reward_model.eval()
        self.logger.info(f"Starting PPO training for {self.epochs} epochs")
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            for step, batch in enumerate(self.dataloader, start=1):
                total_loss += self._train_step(batch)
                if step % self.logging_steps == 0:
                    self.logger.info(f"Epoch {epoch} | Step {step} | Avg PPO Loss: {total_loss/step:.4f}")
            self.logger.info(f"Epoch {epoch} completed | Avg PPO Loss: {total_loss/len(self.dataloader):.4f}")
        self._save_model()

    def _train_step(self, batch):
        input_ids, attention_mask = batch
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        actions = torch.argmax(probs, dim=-1)
        log_probs = torch.log(torch.gather(probs, -1, actions.unsqueeze(-1)).squeeze(-1) + 1e-10)
        rewards = self.reward_model(input_ids, attention_mask=attention_mask).detach()
        advantage = rewards - rewards.mean()
        ratio = torch.exp(log_probs - log_probs.detach())
        clipped = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantage
        ppo_loss = -torch.min(ratio * advantage, clipped).mean()
        self.optimizer.zero_grad()
        ppo_loss.backward()
        self.optimizer.step()
        return ppo_loss.item()

    def _save_model(self):
        out = self.config.get("output", {}).get("model_dir", "models/ppo_model")
        self.model.save_pretrained(out)
        self.tokenizer.save_pretrained(out)
        self.logger.info(f"Saved PPO model to {out}")
