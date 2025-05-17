# training/reward_trainer.py

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from data.data_loader import load_dataset
from RewardModel import RewardModel
from utils.logging_utils import setup_logger

class RewardTrainer:
    """Trainer for reward model training."""
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger("reward")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_id = config.get("model") or config.get("base_model")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(model_id)
        self.model = RewardModel(base_model).to(self.device)

        ds_cfg = config.get("dataset", {})
        self.logger.info(f"Loading dataset with loader: {ds_cfg.get('loader')} name: {ds_cfg.get('name')}")
        dataset = load_dataset(tokenizer=self.tokenizer, dataset_cfg=ds_cfg)
        batch_size = config["training"].get("batch_size", 4)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        lr = config["training"].get("learning_rate", 5e-5)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def train(self):
        epochs = self.config["training"].get("epochs", 3)
        steps = self.config["training"].get("logging_steps", 50)
        self.model.train()
        self.logger.info(f"Training reward model for {epochs} epochs")
        for e in range(1, epochs + 1):
            total_loss = 0.0
            for idx, batch in enumerate(self.dataloader, start=1):
                total_loss += self._train_step(batch)
                if idx % steps == 0:
                    self.logger.info(f"Epoch {e} | Step {idx} | Avg Loss: {total_loss/idx:.4f}")
            self.logger.info(f"Epoch {e} complete | Avg Loss: {total_loss/len(self.dataloader):.4f}")
        self._save_model()

    def _train_step(self, batch):
        inp1, mask1, inp2, mask2, label = batch
        inp1, mask1, inp2, mask2 = [x.to(self.device) for x in (inp1, mask1, inp2, mask2)]
        r1 = self.model(inp1, attention_mask=mask1)
        r2 = self.model(inp2, attention_mask=mask2)
        loss = torch.mean(torch.relu(1 - (r1 - r2)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _save_model(self):
        out = self.config.get("output", {}).get("model_dir", "models/reward_model")
        os.makedirs(out, exist_ok=True)
        torch.save(self.model.state_dict(), f"{out}/reward_model.pth")
        self.logger.info(f"Saved reward model to {out}")
