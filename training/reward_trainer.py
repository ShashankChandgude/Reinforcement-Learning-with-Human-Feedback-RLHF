import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from RewardModel import RewardModel
from DataLoader import load_dolly_dataset
from utils.logging_utils import setup_logger

class RewardTrainer:
    """
    Trainer for reward model training.
    """
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger("reward")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model().to(self.device)
        self.dataloader = self._prepare_dataloader()
        self.optimizer = self._configure_optimizer()

    def _load_tokenizer(self):
        model_id = self.config.get("model") or self.config.get("base_model")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_model(self):
        model_id = self.config.get("model") or self.config.get("base_model")
        base = AutoModelForCausalLM.from_pretrained(model_id)
        return RewardModel(base)

    def _prepare_dataloader(self):
        dc = self.config.get("dataset", {})
        data = load_dolly_dataset(
            tokenizer=self.tokenizer,
            subset_size=dc.get("subset_size", 5000),
            max_length=dc.get("max_seq_length", 512)
        )
        bs = self.config["training"].get("batch_size", 4)
        return DataLoader(data, batch_size=bs, shuffle=True)

    def _configure_optimizer(self):
        lr = self.config["training"].get("learning_rate", 5e-5)
        return torch.optim.AdamW(self.model.parameters(), lr=lr)

    def train(self):
        epochs = self.config["training"].get("epochs", 3)
        steps = self.config["training"].get("logging_steps", 50)
        self.model.train()
        self.logger.info(f"Training reward model for {epochs} epochs")
        for e in range(1, epochs + 1):
            total = 0.0
            for idx, batch in enumerate(self.dataloader, start=1):
                total += self._train_step(batch)
                if idx % steps == 0:
                    self.logger.info(f"Epoch {e} | Step {idx} | Avg Loss: {total / idx:.4f}")
            avg = total / len(self.dataloader)
            self.logger.info(f"Epoch {e} complete | Avg Loss: {avg:.4f}")
        self._save_model()

    def _train_step(self, batch):
        inputs, masks = batch
        inputs, masks = inputs.to(self.device), masks.to(self.device)
        rewards = self.model(inputs, attention_mask=masks)
        loss = -rewards.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _save_model(self):
        out = self.config.get("output", {}).get("model_dir", "models/reward_model")
        torch.save(self.model.state_dict(), f"{out}/reward_model.pth")
        self.logger.info(f"Saved reward model to {out}")
