from data.prepare_sft import prepare_sft_dataset
from data.prepare_reward import prepare_reward_dataset

def load_sft_dataset(tokenizer, dataset_name: str, subset_size: int = 5000, max_length: int = 512):
    return prepare_sft_dataset(
        dataset_name=dataset_name,
        subset_size=subset_size,
        tokenizer=tokenizer,
        max_length=max_length
    )

def load_reward_dataset(tokenizer, dataset_name: str, subset_size: int = 5000, max_length: int = 512):
    return prepare_reward_dataset(
        dataset_name=dataset_name,
        subset_size=subset_size,
        tokenizer=tokenizer,
        max_length=max_length
    )

def load_dataset(tokenizer, dataset_cfg: dict):
    loader_type = dataset_cfg.get("loader")
    if loader_type == "sft":
        return load_sft_dataset(
            tokenizer=tokenizer,
            dataset_name=dataset_cfg.get("name"),
            subset_size=dataset_cfg.get("subset_size", 5000),
            max_length=dataset_cfg.get("max_seq_length", 512)
        )
    elif loader_type == "reward":
        return load_reward_dataset(
            tokenizer=tokenizer,
            dataset_name=dataset_cfg.get("name"),
            subset_size=dataset_cfg.get("subset_size", 5000),
            max_length=dataset_cfg.get("max_seq_length", 512)
        )
    else:
        raise ValueError(f"Unknown loader type: {loader_type}")
