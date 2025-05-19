# tests/test_ood_evaluator.py

import pytest
import torch
from evaluation.ood_evaluator import OODEvaluator
from tests.test_sft_evaluator import DummyTokenizer, DummyModel

@pytest.fixture(autouse=True)
def patch(monkeypatch):
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer,   'from_pretrained', DummyTokenizer.from_pretrained)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', DummyModel.from_pretrained)
    yield

def test_ood_generate_and_metrics(monkeypatch):
    evaluator = OODEvaluator(generation_kwargs={})
    evaluator.load_model('dummy-dir')

    prompts = ['x1', 'x2', 'x3', 'x4']
    preds = evaluator.generate(prompts)
    assert preds == ['decoded_0', 'decoded_1', 'decoded_2', 'decoded_3']

    monkeypatch.setattr('evaluation.ood_evaluator.compute_all', lambda a, b: {'ood': 3.0})
    assert evaluator.compute_metrics(preds, ['r']*4) == {'ood': 3.0}
