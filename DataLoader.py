from datasets import load_dataset
from torch.utils.data import Dataset

class SupervisedDataset(Dataset):
    def __init__(self, prompts, responses, tokenizer, max_length):
        self.prompts = prompts
        self.responses = responses
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        response = self.responses[idx]
        input_text = f"{prompt} {response}"
        tokens = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return tokens["input_ids"].squeeze(), tokens["attention_mask"].squeeze()

def load_dolly_dataset(tokenizer, subset_size=5000, max_length=512):
    dolly_data = load_dataset("databricks/databricks-dolly-15k")
    dolly_subset = dolly_data["train"].select(range(subset_size))
    prompts = [entry["instruction"] for entry in dolly_subset]
    responses = [entry["response"] for entry in dolly_subset]
    return SupervisedDataset(prompts, responses, tokenizer, max_length)
