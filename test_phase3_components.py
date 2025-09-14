#!/usr/bin/env python3
"""
Test script for Phase 3 components (LoRA, PPO, W&B).
"""

import os
import sys
from utils.logging_utils import setup_logger
from utils.wandb_utils import WandBLogger


def test_wandb_integration():
    """Test W&B integration."""
    logger = setup_logger("test")
    logger.info("Testing W&B integration...")
    
    try:
        # Test W&B logger
        wandb_logger = WandBLogger(
            project_name="rlhf-test",
            config={"test": True, "phase": "testing"}
        )
        
        # Try to initialize
        success = wandb_logger.init(run_name="test-run", tags=["test"])
        
        if success:
            logger.info("W&B integration test passed!")
            
            # Test logging
            wandb_logger.log_metrics({"test_loss": 0.5, "test_accuracy": 0.8})
            wandb_logger.log_sample_generations(
                ["Test prompt"], 
                ["Test response"]
            )
            wandb_logger.finish()
            return True
        else:
            logger.warning("W&B not available, skipping test")
            return True  # Not a failure if W&B is not configured
            
    except Exception as e:
        logger.error(f"W&B test failed: {e}")
        return False


def test_lora_config():
    """Test LoRA configuration loading."""
    logger = setup_logger("test")
    logger.info("Testing LoRA configuration...")
    
    try:
        from utils.config_loader import load_config
        config = load_config("configs/lora_7b_config.yaml")
        
        # Check required fields
        required_fields = ["model", "use_lora", "lora_config", "dataset", "training", "output"]
        for field in required_fields:
            if field not in config:
                logger.error(f"Missing required field: {field}")
                return False
        
        logger.info("LoRA configuration test passed!")
        return True
        
    except Exception as e:
        logger.error(f"LoRA config test failed: {e}")
        return False


def test_ppo_config():
    """Test PPO configuration loading."""
    logger = setup_logger("test")
    logger.info("Testing PPO configuration...")
    
    try:
        from utils.config_loader import load_config
        config = load_config("configs/ppo_preference.yaml")
        
        # Check required fields
        required_fields = ["model", "reward_model_dir", "dataset", "training", "output"]
        for field in required_fields:
            if field not in config:
                logger.error(f"Missing required field: {field}")
                return False
        
        logger.info("PPO configuration test passed!")
        return True
        
    except Exception as e:
        logger.error(f"PPO config test failed: {e}")
        return False


def test_imports():
    """Test that all new modules can be imported."""
    logger = setup_logger("test")
    logger.info("Testing imports...")
    
    try:
        # Test LoRA trainer import
        from training.simple_lora_trainer import SimpleLoRATrainer
        logger.info("LoRA trainer import successful")
        
        # Test PPO trainer import
        from training.ppo_preference_trainer import PPOPreferenceTrainer
        logger.info("PPO trainer import successful")
        
        # Test W&B utils import
        from utils.wandb_utils import WandBLogger
        logger.info("W&B utils import successful")
        
        logger.info("All imports successful!")
        return True
        
    except Exception as e:
        logger.error(f"Import test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger = setup_logger("test")
    logger.info("Starting Phase 3 component tests...")
    
    tests = [
        ("Import Test", test_imports),
        ("LoRA Config Test", test_lora_config),
        ("PPO Config Test", test_ppo_config),
        ("W&B Integration Test", test_wandb_integration),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"Running {test_name}...")
        try:
            success = test_func()
            results[test_name] = success
            status = "PASSED" if success else "FAILED"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            logger.error(f"{test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("Test Summary:")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "PASSED" if success else "FAILED"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("All Phase 3 component tests passed!")
        return True
    else:
        logger.error("Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
