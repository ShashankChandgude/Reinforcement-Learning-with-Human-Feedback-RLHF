# data/data_loader.py

from data.prepare_sft import prepare_sft_dataset
from data.prepare_reward import prepare_reward_dataset

def load_sft_dataset(tokenizer, dataset_name: str, subset_size: int = 5000, max_length: int = 512, clean: bool = False):
    return prepare_sft_dataset(
        dataset_name=dataset_name,
        subset_size=subset_size,
        tokenizer=tokenizer,
        max_length=max_length,
        clean=clean
    )

def load_reward_dataset(tokenizer, dataset_name: str, subset_size: int = 5000, max_length: int = 512, clean: bool = False):
    return prepare_reward_dataset(
        dataset_name=dataset_name,
        subset_size=subset_size,
        tokenizer=tokenizer,
        max_length=max_length,
        clean=clean
    )

def load_dataset(tokenizer, dataset_cfg: dict):
    loader = dataset_cfg.get("loader")
    clean = dataset_cfg.get("clean", False)
    name  = dataset_cfg.get("name")
    size  = dataset_cfg.get("subset_size", 5000)
    length= dataset_cfg.get("max_seq_length", 512)

    if loader == "sft":
        return load_sft_dataset(tokenizer, name, size, length, clean=clean)
    if loader == "reward":
        return load_reward_dataset(tokenizer, name, size, length, clean=clean)
    raise ValueError(f"Unknown loader type: {loader}")
