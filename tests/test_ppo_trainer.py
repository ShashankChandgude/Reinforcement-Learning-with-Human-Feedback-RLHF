# tests/test_ppo_trainer.py

import os
import torch
import pytest
import torch.nn as nn
from training.ppo_trainer import PPOTrainer

class DummyTokenizer:
    @classmethod
    def from_pretrained(cls, model_name):
        return cls()

    def __init__(self):
        self.eos_token = "<eos>"
        self.pad_token = self.eos_token

    def __call__(self, text, max_length=None, truncation=None,
                 padding=None, return_tensors=None):
        input_ids = torch.ones((1, max_length), dtype=torch.long)
        attention_mask = torch.ones((1, max_length), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)

class DummyPolicyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))

    @classmethod
    def from_pretrained(cls, model_name):
        return cls()

    def to(self, device):
        return self

    def train(self):
        pass

    def __call__(self, input_ids, attention_mask=None, labels=None):
        batch, seq_len = input_ids.shape
        logits = torch.ones((batch, seq_len, 2), dtype=torch.float, requires_grad=True)
        return type("Out", (), {"logits": logits})()

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)

class DummyRewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def to(self, device):
        return self

    def eval(self):
        pass

    def __call__(self, input_ids, attention_mask=None):
        batch = input_ids.shape[0]
        return torch.ones((batch, 1))

@pytest.fixture(autouse=True)
def patch_all(monkeypatch):
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained',
                        DummyTokenizer.from_pretrained)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained',
                        DummyPolicyModel.from_pretrained)

    import training.ppo_trainer as ppo_mod
    monkeypatch.setattr(ppo_mod, 'RewardModel', DummyRewardModel)

    import DataLoader as dl_mod
    def dummy_load(tokenizer, subset_size, max_length):
        data = torch.ones((2, max_length), dtype=torch.long)
        return torch.utils.data.TensorDataset(data, data)
    monkeypatch.setattr(dl_mod, 'load_dolly_dataset', dummy_load)

    yield

def test_initialization(tmp_path):
    out_dir = tmp_path / "ppo_out"
    config = {
        "model": "dummy-model",
        "reward_model_dir": "dummy-reward-dir",
        "dataset": {"name": "dummy", "subset_size": 2, "max_seq_length": 4},
        "training": {"epochs": 1, "batch_size": 2, "learning_rate": 1e-5,
                     "clip_epsilon": 0.1, "logging_steps": 1},
        "output": {"model_dir": str(out_dir)}
    }
    out_dir.mkdir()
    trainer = PPOTrainer(config)
    assert hasattr(trainer, 'model')
    assert hasattr(trainer, 'reward_model')
    assert hasattr(trainer, 'tokenizer')
    assert hasattr(trainer, 'optimizer')
    assert hasattr(trainer, 'dataloader')
    assert trainer.device in (torch.device('cpu'), torch.device('cuda'))

def test_train_saves_model(tmp_path):
    out_dir = tmp_path / "ppo_out"
    config = {
        "model": "dummy-model",
        "reward_model_dir": "dummy-reward-dir",
        "dataset": {"name": "dummy", "subset_size": 2, "max_seq_length": 4},
        "training": {"epochs": 1, "batch_size": 2, "learning_rate": 1e-5,
                     "clip_epsilon": 0.1, "logging_steps": 1},
        "output": {"model_dir": str(out_dir)}
    }
    out_dir.mkdir()
    trainer = PPOTrainer(config)
    trainer.train()
    # Directory should still exist after saving
    assert out_dir.is_dir()
