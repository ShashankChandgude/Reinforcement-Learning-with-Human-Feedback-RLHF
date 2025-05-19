# tests/test_ppo_evaluator.py

import pytest
import torch
from evaluation.ppo_evaluator import PPOEvaluator
from tests.test_sft_evaluator import DummyTokenizer, DummyModel, patch_transformers

@pytest.fixture(autouse=True)
def patch(monkeypatch):
    # Reuse DummyTokenizer & DummyModel from test_sft_evaluator
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer,   'from_pretrained', DummyTokenizer.from_pretrained)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', DummyModel.from_pretrained)
    yield

def test_ppo_generate_and_metrics(monkeypatch):
    evaluator = PPOEvaluator(generation_kwargs={})
    evaluator.load_model('dummy-dir')

    prompts = ['a', 'b']
    preds = evaluator.generate(prompts)
    assert preds == ['decoded_0', 'decoded_1']

    monkeypatch.setattr('evaluation.ppo_evaluator.compute_all', lambda a, b: {'m': 2.0})
    result = evaluator.compute_metrics(preds, ['x', 'y'])
    assert result == {'m': 2.0}
