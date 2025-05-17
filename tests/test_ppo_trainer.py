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
    def __call__(self, text, max_length=None, truncation=None, padding=None, return_tensors=None):
        return {
            "input_ids": torch.ones((1, max_length), dtype=torch.long),
            "attention_mask": torch.ones((1, max_length), dtype=torch.long)
        }
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
        batch, seq = input_ids.shape
        logits = torch.ones((batch, seq, 2), dtype=torch.float, requires_grad=True)
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
        return torch.ones((input_ids.shape[0], 1))

@pytest.fixture(autouse=True)
def patch_all(monkeypatch, tmp_path):
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer,   'from_pretrained', DummyTokenizer.from_pretrained)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', DummyPolicyModel.from_pretrained)

    import data.data_loader as dl_mod
    from torch.utils.data import TensorDataset
    def dummy_load_sft(tokenizer, dataset_name, subset_size, max_length):
        data = torch.ones((subset_size, max_length), dtype=torch.long)
        return TensorDataset(data, data)
    monkeypatch.setattr(dl_mod, 'load_sft_dataset', dummy_load_sft)

    import training.ppo_trainer as ppo_mod
    monkeypatch.setattr(ppo_mod, 'RewardModel', DummyRewardModel)
    yield

def test_initialization(tmp_path):
    config = {
        "model": "dummy-model",
        "reward_model_dir": "dummy-reward-dir",
        "dataset": {
            "loader": "sft",
            "name": "dummy",
            "subset_size": 2,
            "max_seq_length": 4
        },
        "training": {"epochs": 1, "batch_size": 2, "learning_rate": 1e-5,
                     "clip_epsilon": 0.1, "logging_steps": 1},
        "output": {"model_dir": str(tmp_path / "ppo_out")}
    }
    trainer = PPOTrainer(config)
    assert hasattr(trainer, 'model')
    assert hasattr(trainer, 'reward_model')
    assert hasattr(trainer, 'tokenizer')
    assert hasattr(trainer, 'optimizer')
    assert hasattr(trainer, 'dataloader')

def test_train_saves_model(tmp_path):
    out_dir = tmp_path / "ppo_out"
    config = {
        "model": "dummy-model",
        "reward_model_dir": "dummy-reward-dir",
        "dataset": {
            "loader": "sft",
            "name": "dummy",
            "subset_size": 2,
            "max_seq_length": 4
        },
        "training": {"epochs": 1, "batch_size": 2, "learning_rate": 1e-5,
                     "clip_epsilon": 0.1, "logging_steps": 1},
        "output": {"model_dir": str(out_dir)}
    }
    trainer = PPOTrainer(config)
    trainer.train()
    assert os.path.isdir(str(out_dir))
