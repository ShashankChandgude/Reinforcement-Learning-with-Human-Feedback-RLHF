# tests/test_sft_trainer.py

import os
import torch
import pytest
import torch.nn as nn
from training.sft_trainer import SFTTrainer

class DummyTokenizer:
    eos_token = "<eos>"

    @classmethod
    def from_pretrained(cls, model_name):
        return cls()

    def __call__(self, text, max_length, truncation, padding, return_tensors):
        input_ids = torch.ones((1, max_length), dtype=torch.long)
        attention_mask = torch.ones((1, max_length), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)

class DummyModel(nn.Module):
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
        loss = torch.tensor(0.5, requires_grad=True)
        return type("Out", (), {"loss": loss})()

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)

@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch, tmp_path):
    import transformers
    from torch.utils.data import TensorDataset
    # Patch tokenizer & model
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', DummyTokenizer.from_pretrained)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', DummyModel.from_pretrained)

    import data.data_loader as dl_mod
    def dummy_load_sft(tokenizer, dataset_name, subset_size, max_length, clean=False):
        data = torch.ones((subset_size, max_length), dtype=torch.long)
        return TensorDataset(data, data)
    monkeypatch.setattr(dl_mod, 'load_sft_dataset', dummy_load_sft)
    yield

def test_sft_initialization(tmp_path):
    config = {
        "model": "dummy-model",
        "dataset": {
            "loader": "sft",
            "name": "dummy",
            "subset_size": 2,
            "max_seq_length": 4
        },
        "training": {"epochs": 1, "batch_size": 2, "learning_rate": 1e-5, "logging_steps": 1},
        "output": {"model_dir": str(tmp_path / "model")}
    }
    trainer = SFTTrainer(config)
    assert hasattr(trainer, 'model')
    assert hasattr(trainer, 'tokenizer')
    assert hasattr(trainer, 'optimizer')
    assert hasattr(trainer, 'dataloader')
    assert trainer.device in (torch.device('cpu'), torch.device('cuda'))

def test_sft_train_runs(tmp_path):
    config = {
        "model": "dummy-model",
        "dataset": {
            "loader": "sft",
            "name": "dummy",
            "subset_size": 2,
            "max_seq_length": 4
        },
        "training": {"epochs": 1, "batch_size": 2, "learning_rate": 1e-5, "logging_steps": 1},
        "output": {"model_dir": str(tmp_path / "model")}
    }
    trainer = SFTTrainer(config)
    trainer.train()
    assert os.path.isdir(str(tmp_path / "model"))
