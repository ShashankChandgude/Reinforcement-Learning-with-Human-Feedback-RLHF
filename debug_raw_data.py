"""
Debug script to check raw HH-RLHF data structure.
"""
from datasets import load_dataset

def debug_raw_data():
    """Check the raw HH-RLHF dataset structure."""
    print("=== DEBUGGING RAW HH-RLHF DATA ===\n")
    
    # Load raw dataset
    raw_dataset = load_dataset("Anthropic/hh-rlhf")
    train_data = raw_dataset['train']
    
    print(f"Dataset size: {len(train_data)}")
    print(f"Dataset features: {train_data.features}")
    
    # Check first few examples
    for i in range(3):
        example = train_data[i]
        print(f"\n--- Raw Example {i+1} ---")
        print("Keys:", list(example.keys()))
        
        if 'chosen' in example:
            print(f"CHOSEN (first 200 chars):\n{example['chosen'][:200]}...")
            print(f"CHOSEN (last 200 chars):\n...{example['chosen'][-200:]}")
        
        if 'rejected' in example:
            print(f"REJECTED (first 200 chars):\n{example['rejected'][:200]}...")
            print(f"REJECTED (last 200 chars):\n...{example['rejected'][-200:]}")
        
        print(f"Chosen length: {len(example['chosen'])}")
        print(f"Rejected length: {len(example['rejected'])}")
        print(f"Are they identical? {example['chosen'] == example['rejected']}")

if __name__ == "__main__":
    debug_raw_data()
