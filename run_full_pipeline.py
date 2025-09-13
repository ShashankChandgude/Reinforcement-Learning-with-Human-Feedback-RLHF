#!/usr/bin/env python3
"""
Full RLHF pipeline with LoRA, PPO, and W&B tracking.
"""

import os
import sys
import argparse
from utils.config_loader import load_config
from utils.logging_utils import setup_logger
from utils.wandb_utils import setup_wandb_logging
from training.preference_reward_trainer import PreferenceRewardTrainer
from training.simple_lora_trainer import SimpleLoRATrainer
from training.ppo_preference_trainer import PPOPreferenceTrainer
from evaluation.rlhf_evaluator import run_rlhf_evaluation


def run_reward_model_training(config_path: str, use_wandb: bool = True):
    """Run reward model training."""
    logger = setup_logger("full_pipeline")
    logger.info("Starting reward model training...")
    
    config = load_config(config_path)
    
    # Setup W&B if requested
    wandb_logger = None
    if use_wandb:
        wandb_config = config.copy()
        wandb_config["phase"] = "reward_training"
        wandb_logger = setup_wandb_logging(wandb_config, run_name="reward-model")
    
    try:
        # Train reward model
        trainer = PreferenceRewardTrainer(config)
        trainer.train()
        
        # Log to W&B
        if wandb_logger and wandb_logger.is_initialized:
            wandb_logger.log_model_artifacts(
                config["output"]["model_dir"], 
                "reward-model"
            )
            wandb_logger.finish()
        
        logger.info("Reward model training completed!")
        return True
        
    except Exception as e:
        logger.error(f"Reward model training failed: {e}")
        if wandb_logger:
            wandb_logger.finish()
        return False


def run_lora_training(config_path: str, use_wandb: bool = True):
    """Run LoRA fine-tuning."""
    logger = setup_logger("full_pipeline")
    logger.info("Starting LoRA training...")
    
    config = load_config(config_path)
    
    # Setup W&B if requested
    wandb_logger = None
    if use_wandb:
        wandb_config = config.copy()
        wandb_config["phase"] = "lora_training"
        wandb_logger = setup_wandb_logging(wandb_config, run_name="lora-model")
    
    try:
        # Train with LoRA
        trainer = SimpleLoRATrainer(config)
        trainer.train()
        
        # Log to W&B
        if wandb_logger and wandb_logger.is_initialized:
            wandb_logger.log_model_artifacts(
                config["output"]["model_dir"], 
                "lora-model"
            )
            wandb_logger.finish()
        
        logger.info("LoRA training completed!")
        return True
        
    except Exception as e:
        logger.error(f"LoRA training failed: {e}")
        if wandb_logger:
            wandb_logger.finish()
        return False


def run_ppo_training(config_path: str, use_wandb: bool = True):
    """Run PPO training."""
    logger = setup_logger("full_pipeline")
    logger.info("Starting PPO training...")
    
    config = load_config(config_path)
    
    # Setup W&B if requested
    wandb_logger = None
    if use_wandb:
        wandb_config = config.copy()
        wandb_config["phase"] = "ppo_training"
        wandb_logger = setup_wandb_logging(wandb_config, run_name="ppo-model")
    
    try:
        # Train with PPO
        trainer = PPOPreferenceTrainer(config)
        trainer.train()
        
        # Log training history to W&B
        if wandb_logger and wandb_logger.is_initialized:
            wandb_logger.log_training_curves(trainer.training_history)
            wandb_logger.log_model_artifacts(
                config["output"]["model_dir"], 
                "ppo-model"
            )
            wandb_logger.finish()
        
        logger.info("PPO training completed!")
        return True
        
    except Exception as e:
        logger.error(f"PPO training failed: {e}")
        if wandb_logger:
            wandb_logger.finish()
        return False


def run_evaluation(model_dir: str, use_wandb: bool = True):
    """Run comprehensive evaluation."""
    logger = setup_logger("full_pipeline")
    logger.info("Starting evaluation...")
    
    try:
        # Run evaluation
        results = run_rlhf_evaluation(
            model_dir=model_dir,
            test_prompts=None,  # Will load from dataset
            test_references=None,
            preference_data=None
        )
        
        # Log to W&B
        if use_wandb:
            wandb_config = {"phase": "evaluation", "model_dir": model_dir}
            wandb_logger = setup_wandb_logging(wandb_config, run_name="evaluation")
            
            if wandb_logger.is_initialized:
                # Log evaluation metrics
                if "model_quality" in results:
                    wandb_logger.log_metrics(results["model_quality"])
                
                if "preference_alignment" in results:
                    wandb_logger.log_metrics(results["preference_alignment"])
                
                # Log sample generations
                if "sample_generations" in results:
                    samples = results["sample_generations"]
                    wandb_logger.log_sample_generations(
                        samples["prompts"], 
                        samples["responses"]
                    )
                
                wandb_logger.finish()
        
        logger.info("Evaluation completed!")
        return True
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return False


def run_full_pipeline(config_dir: str = "configs", use_wandb: bool = True):
    """Run the complete RLHF pipeline."""
    logger = setup_logger("full_pipeline")
    logger.info("Starting full RLHF pipeline...")
    
    # Pipeline steps
    steps = [
        ("Reward Model Training", f"{config_dir}/reward_preference.yaml"),
        ("LoRA Training", f"{config_dir}/lora_7b_config.yaml"),
        ("PPO Training", f"{config_dir}/ppo_preference.yaml"),
        ("Evaluation", "models/ppo_preference_model")
    ]
    
    results = {}
    
    for step_name, config_path in steps:
        logger.info(f"Starting {step_name}...")
        
        try:
            if step_name == "Reward Model Training":
                success = run_reward_model_training(config_path, use_wandb)
            elif step_name == "LoRA Training":
                success = run_lora_training(config_path, use_wandb)
            elif step_name == "PPO Training":
                success = run_ppo_training(config_path, use_wandb)
            elif step_name == "Evaluation":
                success = run_evaluation(config_path, use_wandb)
            else:
                logger.error(f"Unknown step: {step_name}")
                success = False
            
            results[step_name] = success
            
            if success:
                logger.info(f"{step_name} completed successfully!")
            else:
                logger.error(f"{step_name} failed!")
                break
                
        except Exception as e:
            logger.error(f"{step_name} failed with exception: {e}")
            results[step_name] = False
            break
    
    # Summary
    logger.info("Pipeline Summary:")
    for step_name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        logger.info(f"  {step_name}: {status}")
    
    all_success = all(results.values())
    if all_success:
        logger.info("Full pipeline completed successfully!")
    else:
        logger.error("Pipeline failed at one or more steps")
    
    return all_success


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run RLHF pipeline")
    parser.add_argument("--phase", choices=["reward", "lora", "ppo", "eval", "full"], 
                       default="full", help="Pipeline phase to run")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--model-dir", type=str, help="Model directory for evaluation")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    
    args = parser.parse_args()
    
    use_wandb = not args.no_wandb
    
    if args.phase == "reward":
        success = run_reward_model_training(args.config or "configs/reward_preference.yaml", use_wandb)
    elif args.phase == "lora":
        success = run_lora_training(args.config or "configs/lora_7b_config.yaml", use_wandb)
    elif args.phase == "ppo":
        success = run_ppo_training(args.config or "configs/ppo_preference.yaml", use_wandb)
    elif args.phase == "eval":
        success = run_evaluation(args.model_dir or "models/ppo_preference_model", use_wandb)
    elif args.phase == "full":
        success = run_full_pipeline(use_wandb=use_wandb)
    else:
        print(f"Unknown phase: {args.phase}")
        success = False
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
