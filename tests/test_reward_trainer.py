# tests/test_reward_trainer.py

import os
import torch
import pytest
import torch.nn as nn
from training.reward_trainer import RewardTrainer

class DummyTokenizer:
    @classmethod
    def from_pretrained(cls, model_name):
        return cls()
    def __init__(self):
        self.eos_token = "<eos>"
    def __call__(self, text, max_length, truncation, padding, return_tensors):
        input_ids = torch.ones((1, max_length), dtype=torch.long)
        attention_mask = torch.ones((1, max_length), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)

class DummyBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("cfg", (), {"hidden_size": 8})()
    @classmethod
    def from_pretrained(cls, model_name):
        return cls()
    def to(self, device):
        return self
    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False):
        batch, seq = input_ids.shape
        hidden = torch.zeros((batch, seq, self.config.hidden_size))
        return type("Out", (), {"hidden_states": [hidden]})()
    def train(self):
        pass

class DummyRewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))
    def to(self, device):
        return self
    def train(self):
        pass
    def forward(self, input_ids, attention_mask=None):
        batch = input_ids.shape[0]
        return torch.ones((batch, 1), requires_grad=True)

@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch, tmp_path):
    import transformers
    from torch.utils.data import TensorDataset
    # Patch tokenizer & base model
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', DummyTokenizer.from_pretrained)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', DummyBaseModel.from_pretrained)

    import data.data_loader as dl_mod
    def dummy_load_reward(tokenizer, dataset_name, subset_size, max_length, clean=False):
        data = torch.ones((subset_size, max_length), dtype=torch.long)
        # five elements needed for reward trainer
        return TensorDataset(data, data, data, data, torch.ones(subset_size, dtype=torch.long))
    monkeypatch.setattr(dl_mod, 'load_reward_dataset', dummy_load_reward)

    import training.reward_trainer as rt_mod
    monkeypatch.setattr(rt_mod, 'RewardModel', DummyRewardModel)
    yield

def test_initialization(tmp_path):
    config = {
        "model": "dummy-model",
        "dataset": {
            "loader": "reward",
            "name": "dummy",
            "subset_size": 2,
            "max_seq_length": 4
        },
        "training": {"epochs": 1, "batch_size": 2, "learning_rate": 1e-5, "logging_steps": 1},
        "output": {"model_dir": str(tmp_path / "model")}
    }
    trainer = RewardTrainer(config)
    assert hasattr(trainer, 'model')
    assert hasattr(trainer, 'tokenizer')
    assert hasattr(trainer, 'optimizer')
    assert hasattr(trainer, 'dataloader')

def test_train_saves_model(tmp_path):
    model_dir = tmp_path / "model"
    config = {
        "model": "dummy-model",
        "dataset": {
            "loader": "reward",
            "name": "dummy",
            "subset_size": 2,
            "max_seq_length": 4
        },
        "training": {"epochs": 1, "batch_size": 2, "learning_rate": 1e-5, "logging_steps": 1},
        "output": {"model_dir": str(model_dir)}
    }
    trainer = RewardTrainer(config)
    trainer.train()
    assert os.path.isdir(str(model_dir))
