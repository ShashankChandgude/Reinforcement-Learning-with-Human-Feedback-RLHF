from datasets import load_dataset
from torch.utils.data import Dataset
import random

def prepare_reward_dataset(dataset_name: str, subset_size: int, tokenizer, max_length: int) -> Dataset:
    """
    Generate a preference dataset for reward model training.
    For each prompt, pairs the original response with a random other response.
    Label 1 indicates the first response is preferred.
    """
    raw = load_dataset(dataset_name)
    split = raw.get("train") or raw.get("validation") or raw[list(raw.keys())[0]]
    subset = split.select(range(min(subset_size, len(split))))
    prompts = subset["instruction"] if "instruction" in subset.column_names else subset["prompt"]
    responses = subset["response"] if "response" in subset.column_names else subset["completion"]

    class RewardDataset(Dataset):
        def __init__(self, prompts, responses, tokenizer, max_length):
            self.items = []
            self.tokenizer = tokenizer
            self.max_length = max_length
            for i, p in enumerate(prompts):
                r1 = responses[i]
                j = random.randrange(len(responses))
                r2 = responses[j]
                self.items.append((p, r1, r2))

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            prompt, r1, r2 = self.items[idx]
            t1 = f"{prompt} {r1}"
            t2 = f"{prompt} {r2}"
            tok1 = self.tokenizer(
                t1,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            tok2 = self.tokenizer(
                t2,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return (
                tok1["input_ids"].squeeze(), tok1["attention_mask"].squeeze(),
                tok2["input_ids"].squeeze(), tok2["attention_mask"].squeeze(),
                1
            )

    return RewardDataset(prompts, responses, tokenizer, max_length)
