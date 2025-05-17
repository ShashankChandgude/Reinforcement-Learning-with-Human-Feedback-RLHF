# notebooks/benchmark_data_loading.py
import time
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from data.data_loader import load_dataset

def benchmark_dataset(
    phase: str,
    model_name: str,
    dataset_cfg: dict,
    batch_size: int = 8,
    num_workers: int = 4,
    iters: int = 100,
):
    """
    phase: 'sft' or 'reward'
    dataset_cfg: {
      'loader': 'sft' / 'reward',
      'name': ...,
      'subset_size': ...,
      'max_seq_length': ...,
      'clean': ...,
      'tokenizer': {...}
    }
    """
    print(f"--- Benchmarking {phase.upper()} loader ---")
    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if not tok.pad_token:
        tok.pad_token = tok.eos_token

    # Dataset build
    t0 = time.time()
    ds = load_dataset(tok, dataset_cfg)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    t1 = time.time()
    print(f"Dataset + DataLoader creation: {t1 - t0:.2f}s for {len(ds)} examples")

    # Iterate a few batches
    t2 = time.time()
    for i, batch in enumerate(loader, 1):
        if i >= iters:
            break
        # move to device and discard
        if torch.cuda.is_available():
            batch = [x.cuda(non_blocking=True) for x in batch]
    t3 = time.time()
    print(f"Iterated {iters} batches of size {batch_size}: {t3 - t2:.2f}s")

if __name__ == "__main__":
    # Example usage for SFT
    sft_cfg = {
        "loader": "sft",
        "name": "databricks/databricks-dolly-15k",
        "subset_size": 5000,
        "max_seq_length": 512,
        "clean": False,
        "tokenizer": {"truncation": True, "padding": "max_length", "return_tensors": "pt"},
    }
    benchmark_dataset(
        phase="sft",
        model_name="EleutherAI/gpt-neo-125M",
        dataset_cfg=sft_cfg,
        batch_size=8,
        num_workers=4,
        iters=50,
    )

    # Example usage for Reward
    reward_cfg = sft_cfg.copy()
    reward_cfg["loader"] = "reward"
    benchmark_dataset(
        phase="reward",
        model_name="EleutherAI/gpt-neo-125M",
        dataset_cfg=reward_cfg,
        batch_size=8,
        num_workers=4,
        iters=50,
    )
