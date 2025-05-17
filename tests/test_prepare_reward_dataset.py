# tests/test_prepare_reward_dataset.py
import torch
import pytest
from datasets import Dataset as HFDataset
from data.prepare_reward import prepare_reward_dataset

class DummyTokenizer:
    def __init__(self):
        self.eos_token = "<eos>"

    def __call__(self, text, max_length, truncation, padding, return_tensors):
        input_ids = torch.arange(max_length, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones((1, max_length), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def save_pretrained(self, save_directory):
        pass

@pytest.fixture(autouse=True)
def patch_load_dataset(monkeypatch):
    data_dict = {"instruction": ["p1", "p2", "p3"], "response": ["r1", "r2", "r3"]}
    hf_ds = HFDataset.from_dict(data_dict)
    import data.prepare_reward as pr_mod
    monkeypatch.setattr(pr_mod, 'load_dataset', lambda name: {'train': hf_ds})
    yield


def test_reward_dataset_length_and_label():
    tokenizer = DummyTokenizer()
    ds = prepare_reward_dataset(
        dataset_name="dummy",
        subset_size=3,
        tokenizer=tokenizer,
        max_length=8
    )
    assert len(ds) == 3
    inp1, m1, inp2, m2, label = ds[0]
    assert inp1.shape == (8,)
    assert m1.shape == (8,)
    assert inp2.shape == (8,)
    assert m2.shape == (8,)
    assert label == 1
