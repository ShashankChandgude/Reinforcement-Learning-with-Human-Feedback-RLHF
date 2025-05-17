# tests/test_prepare_sft_dataset.py
import torch
import pytest
from datasets import Dataset as HFDataset
from data.prepare_sft import prepare_sft_dataset

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
    import data.prepare_sft as ps_mod
    monkeypatch.setattr(ps_mod, 'load_dataset', lambda name: {'train': hf_ds})
    yield


def test_sft_dataset_length_and_shapes():
    tokenizer = DummyTokenizer()
    ds = prepare_sft_dataset(
        dataset_name="dummy",
        subset_size=2,
        tokenizer=tokenizer,
        max_length=10
    )
    assert len(ds) == 2
    input_ids, attn_mask = ds[0]
    assert isinstance(input_ids, torch.Tensor)
    assert input_ids.shape == (10,)
    assert isinstance(attn_mask, torch.Tensor)
    assert attn_mask.shape == (10,)
