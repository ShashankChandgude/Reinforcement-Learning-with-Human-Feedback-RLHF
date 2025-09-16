"""
Simplified LoRA trainer that works reliably.
"""

import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from data.data_loader import load_dataset
from utils.logging_utils import setup_logger
from typing import Dict, Any


class SimpleLoRATrainer:
    """Simplified LoRA trainer that works reliably."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logger("simple_lora_training")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Disable W&B
        os.environ["WANDB_DISABLED"] = "true"
        
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
                bias="none",
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
        
        # Setup data collator
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
    
    def prepare_dataset_for_training(self):
        """Prepare dataset for causal language modeling."""
        self.logger.info("Preparing dataset for training...")
        
        # Convert to HuggingFace format
        texts = []
        for i in range(len(self.dataset)):
            prompt = self.dataset.prompts[i]
            chosen = self.dataset.chosen_responses[i]
            texts.append(f"{prompt} {chosen}")
        
        hf_dataset = Dataset.from_dict({"text": texts})
        
        # Simple tokenization function
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=self.config["dataset"]["max_seq_length"],
                return_tensors=None  # Important: return None for HuggingFace datasets
            )
        
        # Apply tokenization
        tokenized_dataset = hf_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=hf_dataset.column_names
        )
        
        # Split into train/eval
        train_size = int(0.8 * len(tokenized_dataset))
        eval_size = len(tokenized_dataset) - train_size
        
        train_dataset = tokenized_dataset.select(range(train_size))
        eval_dataset = tokenized_dataset.select(range(train_size, train_size + eval_size))
        
        self.logger.info(f"Training dataset size: {len(train_dataset)}")
        self.logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset
    
    def train(self):
        """Train the model with LoRA."""
        train_dataset, eval_dataset = self.prepare_dataset_for_training()
        
        # Setup training arguments with explicit type conversion
        training_args = TrainingArguments(
            output_dir=str(self.config["output"]["model_dir"]),
            logging_dir=str(self.config["output"]["log_dir"]),
            num_train_epochs=int(self.config["training"]["epochs"]),
            per_device_train_batch_size=int(self.config["training"]["batch_size"]),
            per_device_eval_batch_size=int(self.config["training"]["batch_size"]),
            learning_rate=float(self.config["training"]["learning_rate"]),  # Explicit float conversion
            logging_steps=int(self.config["training"]["logging_steps"]),
            save_steps=int(self.config["training"]["save_steps"]),
            eval_steps=int(self.config["training"]["eval_steps"]),
            gradient_accumulation_steps=int(self.config["training"]["gradient_accumulation_steps"]),
            max_grad_norm=float(self.config["training"]["max_grad_norm"]),
            warmup_steps=int(self.config["training"]["warmup_steps"]),
            weight_decay=float(self.config["training"].get("weight_decay", 0.0)),
            lr_scheduler_type=self.config["training"].get("lr_scheduler_type", "linear"),
            save_total_limit=3,  # Keep more checkpoints for better models
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            eval_strategy="steps",
            save_strategy="steps", 
            logging_strategy="steps",
            remove_unused_columns=False,
            report_to=None,  # Disable all reporting including W&B
            dataloader_pin_memory=True,  # Speed optimization
            fp16=False,  # Keep fp32 for GTX 1650 stability
        )
        
        # Print model parameters
        if hasattr(self.model, 'print_trainable_parameters'):
            self.logger.info("Model parameters:")
            self.model.print_trainable_parameters()
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator,
        )
        
        # Start training
        self.logger.info("Starting LoRA training...")
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.logger.info(f"Model saved to {self.config['output']['model_dir']}")
        
        self.logger.info("LoRA training completed successfully!")
        return True
