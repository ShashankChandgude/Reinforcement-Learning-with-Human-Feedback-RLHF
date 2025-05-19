from abc import ABC, abstractmethod
from typing import List, Dict

class Evaluator(ABC):
    """Abstract base class for evaluation routines."""

    @abstractmethod
    def load_model(self, model_dir: str):
        pass

    @abstractmethod
    def generate(self, prompts: List[str]) -> List[str]:
        pass

    @abstractmethod
    def compute_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        pass