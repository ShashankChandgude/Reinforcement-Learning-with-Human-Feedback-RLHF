# training/sft_trainer.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from DataLoader import load_dolly_dataset
from utils.logging_utils import setup_logger


class SFTTrainer:
    """
    Trainer for Supervised Fine-Tuning (SFT) phase.
    Encapsulates model loading, data preparation, training loop, and saving logic.
    """
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger("sft")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = config["model"]
        self.logger.info(f"Loading model and tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        dataset_config = config.get("dataset", {})
        subset_size = dataset_config.get("subset_size", 5000)
        max_length = dataset_config.get("max_seq_length", 512)
        self.logger.info(f"Loading dataset: {dataset_config.get('name')} (subset: {subset_size})")
        dataset = load_dolly_dataset(
            tokenizer=self.tokenizer,
            subset_size=subset_size,
            max_length=max_length
        )
        batch_size = config["training"].get("batch_size", 4)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        lr = config["training"].get("learning_rate", 5e-5)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def train(self):
        epochs = self.config["training"].get("epochs", 3)
        logging_steps = self.config["training"].get("logging_steps", 50)
        self.logger.info(f"Starting SFT training for {epochs} epochs")
        self.model.train()

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for step, (input_ids, attention_mask) in enumerate(self.dataloader, start=1):
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                outputs = self.model(
                    input_ids, attention_mask=attention_mask, labels=input_ids
                )
                loss = outputs.loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                if step % logging_steps == 0:
                    avg_loss = total_loss / step
                    self.logger.info(
                        f"Epoch {epoch} | Step {step} | Avg Loss: {avg_loss:.4f}"
                    )

            avg_epoch_loss = total_loss / len(self.dataloader)
            self.logger.info(f"Epoch {epoch} completed | Avg Loss: {avg_epoch_loss:.4f}")

        output_dir = self.config.get("output", {}).get("model_dir", "models/sft_model")
        self.logger.info(f"Saving model to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.logger.info("SFT training complete")
