#!/usr/bin/env python3
"""
Test the full RLHF pipeline with small-scale configurations.
"""

import os
import sys
import time
from utils.logging_utils import setup_logger
from run_full_pipeline import (
    run_reward_model_training,
    run_lora_training, 
    run_ppo_training,
    run_evaluation
)


def test_reward_model_training():
    """Test reward model training."""
    logger = setup_logger("test_pipeline")
    logger.info("Testing reward model training...")
    
    try:
        success = run_reward_model_training("configs/test_reward_config.yaml", use_wandb=False)
        if success:
            logger.info("[PASS] Reward model training test passed!")
            return True
        else:
            logger.error("[FAIL] Reward model training test failed!")
            return False
    except Exception as e:
        logger.error(f"❌ Reward model training test failed with exception: {e}")
        return False


def test_lora_training():
    """Test LoRA training."""
    logger = setup_logger("test_pipeline")
    logger.info("Testing LoRA training...")
    
    try:
        success = run_lora_training("configs/test_lora_config.yaml", use_wandb=False)
        if success:
            logger.info("[PASS] LoRA training test passed!")
            return True
        else:
            logger.error("[FAIL] LoRA training test failed!")
            return False
    except Exception as e:
        logger.error(f"[FAIL] LoRA training test failed with exception: {e}")
        return False


def test_ppo_training():
    """Test PPO training."""
    logger = setup_logger("test_pipeline")
    logger.info("Testing PPO training...")
    
    try:
        success = run_ppo_training("configs/test_ppo_config.yaml", use_wandb=False)
        if success:
            logger.info("[PASS] PPO training test passed!")
            return True
        else:
            logger.error("[FAIL] PPO training test failed!")
            return False
    except Exception as e:
        logger.error(f"[FAIL] PPO training test failed with exception: {e}")
        return False


def test_evaluation():
    """Test evaluation."""
    logger = setup_logger("test_pipeline")
    logger.info("Testing evaluation...")
    
    try:
        # Test with reward model first
        reward_model_dir = "models/test_reward_model"
        if os.path.exists(reward_model_dir):
            success = run_evaluation(reward_model_dir, use_wandb=False)
            if success:
                logger.info("[PASS] Evaluation test passed!")
                return True
            else:
                logger.error("[FAIL] Evaluation test failed!")
                return False
        else:
            logger.warning("Reward model not found, skipping evaluation test")
            return True
    except Exception as e:
        logger.error(f"[FAIL] Evaluation test failed with exception: {e}")
        return False


def test_full_pipeline():
    """Test the complete pipeline."""
    logger = setup_logger("test_pipeline")
    logger.info("Starting full pipeline test...")
    
    start_time = time.time()
    results = {}
    
    # Test each component
    tests = [
        ("Reward Model Training", test_reward_model_training),
        ("LoRA Training", test_lora_training),
        ("PPO Training", test_ppo_training),
        ("Evaluation", test_evaluation),
    ]
    
    for test_name, test_func in tests:
        logger.info(f"Running {test_name}...")
        test_start = time.time()
        
        try:
            success = test_func()
            test_duration = time.time() - test_start
            results[test_name] = {
                "success": success,
                "duration": test_duration
            }
            
            status = "[PASS]" if success else "[FAIL]"
            logger.info(f"{test_name}: {status} ({test_duration:.1f}s)")
            
            if not success:
                logger.error(f"Pipeline stopped at {test_name}")
                break
                
        except Exception as e:
            test_duration = time.time() - test_start
            logger.error(f"{test_name} failed with exception: {e}")
            results[test_name] = {
                "success": False,
                "duration": test_duration,
                "error": str(e)
            }
            break
    
    # Summary
    total_duration = time.time() - start_time
    logger.info("=" * 60)
    logger.info("PIPELINE TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "[PASS]" if result["success"] else "[FAIL]"
        duration = result["duration"]
        logger.info(f"{test_name}: {status} ({duration:.1f}s)")
        if result["success"]:
            passed += 1
        else:
            if "error" in result:
                logger.error(f"  Error: {result['error']}")
    
    logger.info("-" * 60)
    logger.info(f"Total Tests: {total}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {total - passed}")
    logger.info(f"Total Duration: {total_duration:.1f}s")
    
    if passed == total:
        logger.info("ALL TESTS PASSED! Full pipeline is working!")
        return True
    else:
        logger.error(f"{total - passed} tests failed!")
        return False


def cleanup_test_models():
    """Clean up test models to save space."""
    logger = setup_logger("test_pipeline")
    logger.info("Cleaning up test models...")
    
    test_dirs = [
        "models/test_reward_model",
        "models/test_lora_model", 
        "models/test_ppo_model"
    ]
    
    for dir_path in test_dirs:
        if os.path.exists(dir_path):
            import shutil
            try:
                shutil.rmtree(dir_path)
                logger.info(f"Removed {dir_path}")
            except Exception as e:
                logger.warning(f"Could not remove {dir_path}: {e}")


def main():
    """Main test function."""
    logger = setup_logger("test_pipeline")
    logger.info("Starting full RLHF pipeline test...")
    
    try:
        # Run full pipeline test
        success = test_full_pipeline()
        
        # Ask user if they want to clean up
        if success:
            logger.info("Test completed successfully!")
            response = input("Clean up test models? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                cleanup_test_models()
                logger.info("Test models cleaned up!")
        else:
            logger.error("Test failed! Keeping models for debugging.")
        
        return success
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
