"""
Debug script to check reward model direction and data alignment.
"""
import torch
import yaml
from training.preference_reward_trainer import PreferenceRewardTrainer
from data.data_loader import load_dataset
from transformers import AutoTokenizer

def debug_reward_direction():
    """Debug the reward model training direction."""
    print("=== DEBUGGING REWARD MODEL DIRECTION ===\n")
    
    # Load config
    with open("configs/reward_preference.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Load tokenizer
    model_id = config.get("model") or config.get("base_model")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    ds_cfg = config.get("dataset", {})
    dataset = load_dataset(tokenizer=tokenizer, dataset_cfg=ds_cfg)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"First few examples:")
    
    # Check first 3 examples
    for i in range(min(3, len(dataset))):
        item = dataset[i]
        print(f"\n--- Example {i+1} ---")
        print(f"PROMPT: {item['prompt'][:100]}...")
        print(f"CHOSEN: {item['chosen_text'][:100]}...")
        print(f"REJECTED: {item['rejected_text'][:100]}...")
        
        # Check if chosen is actually better
        print(f"Chosen length: {len(item['chosen_text'])}")
        print(f"Rejected length: {len(item['rejected_text'])}")
    
    # Now test with a trained model
    trainer = PreferenceRewardTrainer(config)
    
    # Load the latest trained model if it exists
    import os
    model_path = "models/reward_model_preference"
    if os.path.exists(f"{model_path}/reward_model.pth"):
        print(f"\nLoading trained model from {model_path}")
        trainer.model.load_state_dict(torch.load(f"{model_path}/reward_model.pth", map_location='cpu'))
        trainer.model.eval()
    
    # Test on first batch
    from torch.utils.data import DataLoader
    test_loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=trainer.collate_fn)
    batch = next(iter(test_loader))
    
    print(f"\n=== TESTING BATCH COMPUTATION ===")
    print(f"Batch size: {len(batch['prompts'])}")
    
    with torch.no_grad():
        chosen_input_ids = batch['chosen_input_ids'].to(trainer.device)
        chosen_attention_mask = batch['chosen_attention_mask'].to(trainer.device)
        rejected_input_ids = batch['rejected_input_ids'].to(trainer.device)
        rejected_attention_mask = batch['rejected_attention_mask'].to(trainer.device)
        
        # Get rewards
        chosen_rewards = trainer.model(chosen_input_ids, attention_mask=chosen_attention_mask)
        rejected_rewards = trainer.model(rejected_input_ids, attention_mask=rejected_attention_mask)
        
        print(f"Chosen rewards shape: {chosen_rewards.shape}")
        print(f"Rejected rewards shape: {rejected_rewards.shape}")
        print(f"Chosen rewards: {chosen_rewards.squeeze().cpu().numpy()}")
        print(f"Rejected rewards: {rejected_rewards.squeeze().cpu().numpy()}")
        
        # Compute difference
        diff = (chosen_rewards - rejected_rewards).squeeze().cpu().numpy()
        print(f"Reward differences (chosen - rejected): {diff}")
        print(f"Mean difference: {diff.mean():.4f}")
        print(f"Positive differences: {(diff > 0).sum()}/{len(diff)}")
        print(f"Negative differences: {(diff < 0).sum()}/{len(diff)}")
        
        # Check accuracy calculation
        pred = (chosen_rewards > rejected_rewards).cpu().numpy()
        accuracy = pred.mean()
        print(f"Preference accuracy: {accuracy:.3f}")
        
        # Test loss computation
        loss = trainer.compute_preference_loss(chosen_rewards, rejected_rewards)
        print(f"Loss: {loss.item():.4f}")
        
        # Check if we're using margin
        print(f"Using margin: {trainer.margin}")
        
        # Manual loss computation for verification
        reward_diff = (chosen_rewards - rejected_rewards).squeeze()
        if trainer.margin > 0.0:
            manual_loss = torch.relu(trainer.margin - reward_diff).mean()
            print(f"Manual hinge loss: {manual_loss.item():.4f}")
        else:
            targets = torch.ones_like(reward_diff)
            manual_loss = torch.nn.functional.binary_cross_entropy_with_logits(reward_diff, targets)
            print(f"Manual BCE loss: {manual_loss.item():.4f}")

if __name__ == "__main__":
    debug_reward_direction()