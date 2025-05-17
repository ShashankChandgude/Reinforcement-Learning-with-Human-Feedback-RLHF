import os
import torch
import pytest
from training.sft_trainer import SFTTrainer

# Dummy classes to simulate dependencies
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

import torch.nn as nn
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Add a dummy parameter to avoid empty parameter list
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
        return type("Output", (), {"loss": loss})()

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch, tmp_path):
    # Patch transformers
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', DummyTokenizer.from_pretrained)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', DummyModel.from_pretrained)

    # Patch dataset loader
    import DataLoader as dl_mod
    def dummy_load(tokenizer, subset_size, max_length):
        data = torch.ones((2, max_length), dtype=torch.long)
        return torch.utils.data.TensorDataset(data, data)
    monkeypatch.setattr(dl_mod, 'load_dolly_dataset', dummy_load)
    yield


def test_sft_initialization(tmp_path):
    config = {
        "model": "dummy-model",
        "dataset": {"name": "dummy", "subset_size": 2, "max_seq_length": 4},
        "training": {"epochs": 1, "batch_size": 2, "learning_rate": 1e-5, "logging_steps": 1},
        "output": {"model_dir": str(tmp_path / "model_dir")}
    }
    trainer = SFTTrainer(config)
    assert hasattr(trainer, 'model')
    assert hasattr(trainer, 'tokenizer')
    assert trainer.device in (torch.device('cpu'), torch.device('cuda'))
    assert len(trainer.dataloader) == 1


def test_sft_train_runs(tmp_path):
    config = {
        "model": "dummy-model",
        "dataset": {"name": "dummy", "subset_size": 2, "max_seq_length": 4},
        "training": {"epochs": 1, "batch_size": 2, "learning_rate": 1e-5, "logging_steps": 1},
        "output": {"model_dir": str(tmp_path / "model_dir")}
    }
    trainer = SFTTrainer(config)
    trainer.train()
    assert os.path.isdir(str(tmp_path / "model_dir"))