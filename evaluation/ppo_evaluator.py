# evaluation/ppo_evaluator.py

from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluation.evaluator import Evaluator
from evaluation.eval_metrics import compute_all

class PPOEvaluator(Evaluator):
    """Evaluator for models fine-tuned via Proximal Policy Optimization."""
    def __init__(self, generation_kwargs: Dict = None):
        # Default generation strategy; can be overridden
        self.generation_kwargs = generation_kwargs or {
            "max_length": 128,
            "num_beams": 5,
            "early_stopping": True
        }

    def load_model(self, model_dir: str):
        # Load tokenizer and model from directory
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.model.eval()

    def generate(self, prompts: List[str]) -> List[str]:
        # Tokenize inputs with padding/truncation
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **self.generation_kwargs
        )
        # Decode predictions
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def compute_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        # Compute BLEU, ROUGE, BERTScore
        return compute_all(predictions, references)
