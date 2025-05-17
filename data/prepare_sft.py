# data/prepare_sft.py

from datasets import load_dataset
from torch.utils.data import Dataset
from utils.text_cleaner import clean_text

def prepare_sft_dataset(
    dataset_name: str,
    subset_size: int,
    tokenizer,
    max_length: int,
    clean: bool = False
) -> Dataset:
    raw = load_dataset(dataset_name)
    split = raw.get("train") or raw.get("validation") or raw[list(raw.keys())[0]]
    subset = split.select(range(min(subset_size, len(split))))
    prompts = subset["instruction"] if "instruction" in subset.column_names else subset["prompt"]
    responses = subset["response"] if "response" in subset.column_names else subset["completion"]

    class SFTDataset(Dataset):
        def __init__(self, prompts, responses, tokenizer, max_length, clean):
            self.prompts = prompts
            self.responses = responses
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.clean = clean

        def __len__(self):
            return len(self.prompts)

        def __getitem__(self, idx):
            p = self.prompts[idx]
            r = self.responses[idx]
            if self.clean:
                p = clean_text(p)
                r = clean_text(r)
            text = f"{p} {r}"
            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return tokens["input_ids"].squeeze(), tokens["attention_mask"].squeeze()

    return SFTDataset(prompts, responses, tokenizer, max_length, clean)
