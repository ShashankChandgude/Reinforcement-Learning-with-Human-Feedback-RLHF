from datasets import load_dataset
from torch.utils.data import Dataset

def prepare_sft_dataset(dataset_name: str, subset_size: int, tokenizer, max_length: int) -> Dataset:
    """
    Load a supervised fine-tuning dataset from HuggingFace and return a PyTorch Dataset.
    Each example concatenates prompt and response for tokenization.
    """
    raw = load_dataset(dataset_name)
    split = raw.get("train") or raw.get("validation") or raw[list(raw.keys())[0]]
    subset = split.select(range(min(subset_size, len(split))))
    prompts = subset["instruction"] if "instruction" in subset.column_names else subset["prompt"]
    responses = subset["response"] if "response" in subset.column_names else subset["completion"]

    class SFTDataset(Dataset):
        def __init__(self, prompts, responses, tokenizer, max_length):
            self.prompts = prompts
            self.responses = responses
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.prompts)

        def __getitem__(self, idx):
            text = f"{self.prompts[idx]} {self.responses[idx]}"
            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return tokens["input_ids"].squeeze(), tokens["attention_mask"].squeeze()

    return SFTDataset(prompts, responses, tokenizer, max_length)
