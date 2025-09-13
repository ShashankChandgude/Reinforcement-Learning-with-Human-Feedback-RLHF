#!/usr/bin/env python3
"""
Small-scale training script to verify the RLHF pipeline works.
This runs a minimal training to test the complete pipeline.
"""

import torch
from transformers import AutoTokenizer
from training.preference_reward_trainer import PreferenceRewardTrainer
from utils.config_loader import load_config
from utils.logging_utils import setup_logger

def run_small_training():
    """Run a small-scale training to test the pipeline."""
    logger = setup_logger("small_training")
    logger.info("Starting small-scale RLHF training test")
    
    # Load configuration
    config = load_config("configs/reward_preference.yaml")
    
    # Reduce dataset size for quick testing
    config["dataset"]["subset_size"] = 50  # Very small for testing
    config["training"]["epochs"] = 1  # Single epoch
    config["training"]["batch_size"] = 2  # Small batch size
    config["training"]["logging_steps"] = 10  # More frequent logging
    
    logger.info(f"Configuration: {config}")
    
    try:
        # Initialize trainer
        trainer = PreferenceRewardTrainer(config)
        
        # Run training
        logger.info("Starting preference reward model training...")
        trainer.train()
        
        logger.info("✅ Small-scale training completed successfully!")
        
        # Test evaluation
        logger.info("Testing preference accuracy...")
        from torch.utils.data import DataLoader
        test_dataloader = DataLoader(
            trainer.dataset, 
            batch_size=2, 
            shuffle=False,
            collate_fn=trainer.collate_fn
        )
        
        accuracy = trainer.evaluate_preference_accuracy(test_dataloader)
        logger.info(f"Final preference accuracy: {accuracy:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_small_training()
    exit(0 if success else 1)
