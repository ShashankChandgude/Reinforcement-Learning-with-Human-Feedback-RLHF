# tests/test_sft_evaluator.py

import pytest
import torch
from evaluation.sft_evaluator import SFTEvaluator

class DummyTokenizer:
    @classmethod
    def from_pretrained(cls, model_dir):
        return cls()
    def __init__(self):
        self.eos_token = '<eos>'
        self.pad_token = '<pad>'
    def __call__(self, texts, return_tensors, padding, truncation):
        batch_size = len(texts)
        return {
            'input_ids': torch.zeros((batch_size, 1), dtype=torch.long),
            'attention_mask': torch.ones((batch_size, 1), dtype=torch.long)
        }
    def batch_decode(self, outputs, skip_special_tokens):
        return [f'decoded_{i}' for i in range(outputs.size(0))]

class DummyModel(torch.nn.Module):
    @classmethod
    def from_pretrained(cls, model_dir):
        return cls()
    def eval(self): pass
    def generate(self, input_ids, attention_mask, **kwargs):
        batch_size = input_ids.size(0)
        return torch.arange(batch_size).unsqueeze(-1)

@pytest.fixture(autouse=True)
def patch_transformers(monkeypatch):
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer,   'from_pretrained', DummyTokenizer.from_pretrained)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', DummyModel.from_pretrained)
    yield

def test_sft_generate_and_metrics(monkeypatch):
    evaluator = SFTEvaluator(generation_kwargs={})
    evaluator.load_model('dummy-dir')

    prompts = ['p1', 'p2', 'p3']
    preds = evaluator.generate(prompts)
    assert preds == ['decoded_0', 'decoded_1', 'decoded_2']

    monkeypatch.setattr('evaluation.sft_evaluator.compute_all', lambda a, b: {'test': 1.0})
    metrics = evaluator.compute_metrics(preds, ['r1', 'r2', 'r3'])
    assert metrics == {'test': 1.0}
