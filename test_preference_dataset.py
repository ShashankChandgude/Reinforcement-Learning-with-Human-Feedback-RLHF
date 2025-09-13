#!/usr/bin/env python3
"""
Test script to verify preference dataset loading works correctly.
"""

import torch
from transformers import AutoTokenizer
from data.data_loader import load_dataset

def test_preference_dataset():
    """Test loading and processing of preference dataset."""
    print("Testing preference dataset loading...")
    
    # Load a small model for testing
    model_name = "EleutherAI/gpt-neo-125M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Dataset configuration
    dataset_cfg = {
        "loader": "preference",
        "name": "Anthropic/hh-rlhf",
        "subset_size": 10,  # Very small for testing
        "max_seq_length": 256,
        "clean": True,
        "tokenizer": {
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "pt"
        }
    }
    
    try:
        # Load dataset
        dataset = load_dataset(tokenizer, dataset_cfg)
        print(f"✓ Dataset loaded successfully with {len(dataset)} examples")
        
        # Test a few examples
        for i in range(min(3, len(dataset))):
            example = dataset[i]
            print(f"\n--- Example {i+1} ---")
            print(f"Prompt: {example['prompt'][:100]}...")
            print(f"Chosen: {example['chosen_text'][:100]}...")
            print(f"Rejected: {example['rejected_text'][:100]}...")
            print(f"Chosen input shape: {example['chosen_input_ids'].shape}")
            print(f"Rejected input shape: {example['rejected_input_ids'].shape}")
        
        # Test DataLoader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        batch = next(iter(dataloader))
        print(f"\n✓ DataLoader works - batch size: {batch['chosen_input_ids'].shape[0]}")
        print(f"Chosen batch shape: {batch['chosen_input_ids'].shape}")
        print(f"Rejected batch shape: {batch['rejected_input_ids'].shape}")
        
        print("\n✅ All tests passed! Preference dataset is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error loading preference dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_preference_dataset()
    exit(0 if success else 1)
