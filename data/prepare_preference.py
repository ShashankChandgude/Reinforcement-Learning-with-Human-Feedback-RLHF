"""
Preference dataset preparation for RLHF training.
Supports Anthropic HH-RLHF and other preference datasets.
"""

from datasets import load_dataset
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any
import random


class PreferenceDataset(Dataset):
    """Dataset for preference-based training (reward model, PPO)."""
    
    def __init__(self, 
                 prompts: List[str], 
                 chosen_responses: List[str], 
                 rejected_responses: List[str],
                 tokenizer,
                 max_length: int = 512,
                 clean: bool = False,
                 tokenizer_kwargs: dict = None):
        """
        Args:
            prompts: List of input prompts
            chosen_responses: List of preferred responses
            rejected_responses: List of non-preferred responses
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            clean: Whether to clean text
            tokenizer_kwargs: Additional tokenizer arguments
        """
        self.prompts = prompts
        self.chosen_responses = chosen_responses
        self.rejected_responses = rejected_responses
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.clean = clean
        
        # Set up tokenizer kwargs
        defaults = {
            "truncation": True, 
            "padding": "max_length", 
            "return_tensors": "pt"
        }
        self.tokenizer_kwargs = defaults
        if tokenizer_kwargs:
            self.tokenizer_kwargs.update(tokenizer_kwargs)
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        chosen = self.chosen_responses[idx]
        rejected = self.rejected_responses[idx]
        
        if self.clean:
            from utils.text_cleaner import clean_text
            prompt = clean_text(prompt)
            chosen = clean_text(chosen)
            rejected = clean_text(rejected)
        
        # Tokenize chosen response
        chosen_text = f"{prompt} {chosen}"
        chosen_tokens = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            **self.tokenizer_kwargs
        )
        
        # Tokenize rejected response
        rejected_text = f"{prompt} {rejected}"
        rejected_tokens = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            **self.tokenizer_kwargs
        )
        
        return {
            'prompt': prompt,
            'chosen_input_ids': chosen_tokens["input_ids"].squeeze(),
            'chosen_attention_mask': chosen_tokens["attention_mask"].squeeze(),
            'rejected_input_ids': rejected_tokens["input_ids"].squeeze(),
            'rejected_attention_mask': rejected_tokens["attention_mask"].squeeze(),
            'chosen_text': chosen,
            'rejected_text': rejected
        }


def prepare_hh_rlhf_dataset(
    dataset_name: str = "Anthropic/hh-rlhf",
    subset_size: int = 1000,
    tokenizer=None,
    max_length: int = 512,
    clean: bool = False,
    tokenizer_kwargs: dict = None
) -> PreferenceDataset:
    """
    Prepare Anthropic HH-RLH dataset for preference training.
    
    Args:
        dataset_name: Name of the dataset (default: Anthropic/hh-rlhf)
        subset_size: Number of examples to use
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
        clean: Whether to clean text
        tokenizer_kwargs: Additional tokenizer arguments
    
    Returns:
        PreferenceDataset for training
    """
    print(f"Loading dataset: {dataset_name}")
    raw_dataset = load_dataset(dataset_name)
    
    # Use the 'train' split
    train_data = raw_dataset['train']
    
    # Take subset
    if subset_size > len(train_data):
        subset_size = len(train_data)
        print(f"Warning: subset_size larger than dataset, using all {subset_size} examples")
    
    # Select subset
    subset = train_data.select(range(subset_size))
    
    # Extract prompts and responses
    prompts = []
    chosen_responses = []
    rejected_responses = []
    
    for example in subset:
        # HH-RLHF has 'chosen' and 'rejected' fields
        if 'chosen' in example and 'rejected' in example:
            chosen_text = example['chosen'].strip()
            rejected_text = example['rejected'].strip()
            
            # For HH-RLHF, extract the FIRST human query and use FULL conversations as responses
            # The dataset contains full multi-turn conversations
            
            # Extract first human prompt - handle the \n\nH: format
            if 'H:' in chosen_text:
                # Find first H: occurrence
                h_start = chosen_text.find('H:')
                if h_start != -1:
                    # Extract the line with H:
                    h_line_start = h_start
                    h_line_end = chosen_text.find('\n', h_start)
                    if h_line_end == -1:
                        h_line_end = len(chosen_text)
                    
                    # Get the human prompt
                    prompt = chosen_text[h_start+2:h_line_end].strip()
                    
                    # Get everything after the first H: line as the conversation
                    chosen_response = chosen_text[h_line_end:].strip()
                else:
                    continue
            else:
                # Fallback for non-standard format
                prompt = "Continue the conversation:"
                chosen_response = chosen_text
            
            # Extract rejected response (same prompt, different conversation)
            if 'H:' in rejected_text:
                h_start = rejected_text.find('H:')
                if h_start != -1:
                    h_line_end = rejected_text.find('\n', h_start)
                    if h_line_end == -1:
                        h_line_end = len(rejected_text)
                    rejected_response = rejected_text[h_line_end:].strip()
                else:
                    continue
            else:
                rejected_response = rejected_text
            
            # Skip if responses are identical or empty
            if chosen_response == rejected_response or not chosen_response or not rejected_response:
                continue
                
            prompts.append(prompt)
            chosen_responses.append(chosen_response)
            rejected_responses.append(rejected_response)
    
    print(f"Prepared {len(prompts)} preference pairs")
    
    return PreferenceDataset(
        prompts=prompts,
        chosen_responses=chosen_responses,
        rejected_responses=rejected_responses,
        tokenizer=tokenizer,
        max_length=max_length,
        clean=clean,
        tokenizer_kwargs=tokenizer_kwargs
    )


def prepare_preference_dataset(
    dataset_name: str,
    subset_size: int = 1000,
    tokenizer=None,
    max_length: int = 512,
    clean: bool = False,
    tokenizer_kwargs: dict = None
) -> PreferenceDataset:
    """
    Generic preference dataset preparation.
    Currently supports HH-RLHF, can be extended for other datasets.
    """
    if "hh-rlhf" in dataset_name.lower():
        return prepare_hh_rlhf_dataset(
            dataset_name=dataset_name,
            subset_size=subset_size,
            tokenizer=tokenizer,
            max_length=max_length,
            clean=clean,
            tokenizer_kwargs=tokenizer_kwargs
        )
    else:
        raise ValueError(f"Unsupported preference dataset: {dataset_name}")
