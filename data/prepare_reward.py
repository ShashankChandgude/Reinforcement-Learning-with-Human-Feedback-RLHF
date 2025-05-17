from datasets import load_dataset
from torch.utils.data import Dataset
from utils.text_cleaner import clean_text
import random

def prepare_reward_dataset(
    dataset_name: str,
    subset_size: int,
    tokenizer,
    max_length: int,
    clean: bool = False,
    tokenizer_kwargs: dict = None
) -> Dataset:
    raw = load_dataset(dataset_name)
    split = raw.get("train") or raw.get("validation") or raw[list(raw.keys())[0]]
    subset = split.select(range(min(subset_size, len(split))))
    prompts = subset["instruction"] if "instruction" in subset.column_names else subset["prompt"]
    responses = subset["response"] if "response" in subset.column_names else subset["completion"]

    class RewardDataset(Dataset):
        def __init__(self, prompts, responses, tokenizer, max_length, clean, tokenizer_kwargs):
            self.items = []
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.clean = clean
            defaults = {"truncation": True, "padding": "max_length", "return_tensors": "pt"}
            self.tokenizer_kwargs = defaults
            if tokenizer_kwargs:
                self.tokenizer_kwargs.update(tokenizer_kwargs)

            for i, p in enumerate(prompts):
                r1 = responses[i]
                r2 = responses[random.randrange(len(responses))]
                self.items.append((p, r1, r2))

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            p, r1, r2 = self.items[idx]
            if self.clean:
                p = clean_text(p)
                r1 = clean_text(r1)
                r2 = clean_text(r2)
            t1 = f"{p} {r1}"
            t2 = f"{p} {r2}"
            tok1 = self.tokenizer(t1, max_length=self.max_length, **self.tokenizer_kwargs)
            tok2 = self.tokenizer(t2, max_length=self.max_length, **self.tokenizer_kwargs)
            return (
                tok1["input_ids"].squeeze(), tok1["attention_mask"].squeeze(),
                tok2["input_ids"].squeeze(), tok2["attention_mask"].squeeze(),
                1
            )

    return RewardDataset(prompts, responses, tokenizer, max_length, clean, tokenizer_kwargs)
