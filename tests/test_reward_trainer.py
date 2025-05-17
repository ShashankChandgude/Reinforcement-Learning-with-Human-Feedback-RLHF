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
        batch_size, seq_len = input_ids.shape
        hidden = torch.zeros((batch_size, seq_len, self.config.hidden_size))
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
        batch_size = input_ids.shape[0]
        # Ensure tensor requires grad so loss.backward() works
        return torch.ones((batch_size, 1), requires_grad=True)

@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    import transformers
    from training.reward_trainer import RewardModel as OriginalRM

    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', DummyTokenizer.from_pretrained)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', DummyBaseModel.from_pretrained)

    import DataLoader as dl_mod
    def dummy_load(tokenizer, subset_size, max_length):
        data = torch.ones((2, max_length), dtype=torch.long)
        return torch.utils.data.TensorDataset(data, data)
    monkeypatch.setattr(dl_mod, 'load_dolly_dataset', dummy_load)

    monkeypatch.setattr('training.reward_trainer.RewardModel', DummyRewardModel)
    yield

def test_initialization(tmp_path):
    model_dir = tmp_path / "model_out"
    config = {
        "model": "dummy-model",
        "dataset": {"name": "dummy", "subset_size": 2, "max_seq_length": 4},
        "training": {"epochs": 1, "batch_size": 2, "learning_rate": 1e-5, "logging_steps": 1},
        "output": {"model_dir": str(model_dir)}
    }
    model_dir.mkdir()
    trainer = RewardTrainer(config)
    assert hasattr(trainer, 'model')
    assert hasattr(trainer, 'tokenizer')
    assert hasattr(trainer, 'optimizer')
    assert hasattr(trainer, 'dataloader')
    assert trainer.device in (torch.device('cpu'), torch.device('cuda'))

def test_train_saves_model(tmp_path):
    model_dir = tmp_path / "model_out"
    config = {
        "model": "dummy-model",
        "dataset": {"name": "dummy", "subset_size": 2, "max_seq_length": 4},
        "training": {"epochs": 1, "batch_size": 2, "learning_rate": 1e-5, "logging_steps": 1},
        "output": {"model_dir": str(model_dir)}
    }
    model_dir.mkdir()
    trainer = RewardTrainer(config)
    trainer.train()
    saved_file = model_dir / "reward_model.pth"
    assert saved_file.is_file()
