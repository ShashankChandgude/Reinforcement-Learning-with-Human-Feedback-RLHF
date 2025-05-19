# evaluation/ood_evaluator.py

from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluation.evaluator import Evaluator
from evaluation.eval_metrics import compute_all

class OODEvaluator(Evaluator):
    """Evaluator for Out-of-Distribution datasets."""
    def __init__(self, generation_kwargs: Dict = None):
        self.generation_kwargs = generation_kwargs or {
            "max_length": 128,
            "num_beams": 5,
            "early_stopping": True
        }

    def load_model(self, model_dir: str):
        # Load tokenizer and model for OOD evaluation
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.model.eval()

    def generate(self, prompts: List[str]) -> List[str]:
        # Tokenize prompts with padding and truncation
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        # Generate outputs using same generation settings
        outputs = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            **self.generation_kwargs
        )
        # Decode to text
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def compute_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        # Use facade to compute BLEU, ROUGE, BERTScore
        return compute_all(predictions, references)
