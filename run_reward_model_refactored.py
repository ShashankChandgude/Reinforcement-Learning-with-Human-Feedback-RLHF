#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dedicated Reward Model Training Script (Refactored)
Runs the refactored balanced reward trainer independently for testing and development.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any

# ---- Runtime env knobs (helps Windows + 4GB cards) ----
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import refactored trainer
from training.Reward_trainer.refactored_balanced_trainer import RefactoredBalancedRewardTrainer
from utils.config_loader import load_config
from utils.logging_utils import setup_logger


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Refactored Reward Model Training - Dedicated Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_reward_model_refactored.py                                    # Use default config
  python run_reward_model_refactored.py --config configs/custom_config.yaml  # Use custom config
  python run_reward_model_refactored.py --force                            # Force retrain even if model exists
  python run_reward_model_refactored.py --epochs 10                       # Override epochs
        """
    )
    
    parser.add_argument(
        '--config',
        default='configs/reward_preference_balanced.yaml',
        help='Path to configuration file (default: configs/reward_preference_balanced.yaml)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force retraining even if model already exists'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        help='Override number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Override batch size'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        help='Override learning rate'
    )
    
    parser.add_argument(
        '--output-dir',
        help='Override output directory for the trained model'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def check_model_exists(output_dir: str) -> bool:
    """Check if model already exists in the output directory."""
    if not os.path.exists(output_dir):
        return False
    
    # Check for key model files
    model_files = [
        "reward_model.pth",
        "config.json",
        "model.safetensors"
    ]
    
    for file in model_files:
        if os.path.exists(os.path.join(output_dir, file)):
            return True
    
    return False


def print_banner(title: str, subtitle: str = "", width: int = 80):
    """Print a formatted banner."""
    print("=" * width)
    print(title)
    if subtitle:
        print(subtitle)
    print("=" * width)


def print_results(success: bool, metrics: Dict[str, Any], training_time: float):
    """Print training results."""
    status = "SUCCESS" if success else "FAILED"
    print(f"\n{status}: Refactored Reward Model Training")
    print("-" * 50)
    
    if success:
        print(f"Training Time: {training_time/60:.1f} minutes")
        print(f"Epochs Completed: {metrics.get('epochs_completed', 0)}")
        print(f"Training Completed: {metrics.get('training_completed', False)}")
        
        # Print accuracy prominently
        final_accuracy = metrics.get('final_accuracy', 0.0)
        print(f"🎯 Final Accuracy: {final_accuracy:.2f}%")
        
        # Print final metrics if available
        final_metrics = metrics.get('final_metrics', {})
        if final_metrics:
            print("\nDetailed Metrics:")
            for key, value in final_metrics.items():
                if isinstance(value, float):
                    if 'accuracy' in key.lower():
                        print(f"  • {key}: {value:.2f}%")
                    else:
                        print(f"  • {key}: {value:.4f}")
                else:
                    print(f"  • {key}: {value}")
        
        print(f"Model Path: {metrics.get('model_path', 'N/A')}")
    else:
        print(f"Error: {metrics.get('error', 'Unknown error')}")
    
    print("-" * 50)


def main():
    """Main function to run refactored reward model training."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logger("refactored_reward_training")
    
    print_banner(
        "REFACTORED REWARD MODEL TRAINING",
        "Advanced monitoring and error handling with preserved optimizations",
        width=80
    )
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Override configuration with command line arguments
        if args.epochs:
            config['training']['epochs'] = args.epochs
            logger.info(f"Overriding epochs: {args.epochs}")
        
        if args.batch_size:
            config['dataset']['batch_size'] = args.batch_size
            config['training']['batch_size'] = args.batch_size
            logger.info(f"Overriding batch size: {args.batch_size}")
        
        if args.learning_rate:
            config['training']['learning_rate'] = args.learning_rate
            logger.info(f"Overriding learning rate: {args.learning_rate}")
        
        if args.output_dir:
            config['output']['model_dir'] = args.output_dir
            logger.info(f"Overriding output directory: {args.output_dir}")
        
        # Check if model already exists
        output_dir = config['output']['model_dir']
        if check_model_exists(output_dir) and not args.force:
            logger.info(f"Model already exists at: {output_dir}")
            print(f"\nModel already exists at: {output_dir}")
            print("Use --force to retrain anyway")
            return True
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Print configuration summary
        print(f"\nConfiguration Summary:")
        print(f"  • Model: {config.get('model', 'N/A')}")
        print(f"  • Dataset: {config['dataset'].get('name', 'N/A')}")
        print(f"  • Epochs: {config['training'].get('epochs', 'N/A')}")
        print(f"  • Batch Size: {config['dataset'].get('batch_size', 'N/A')}")
        print(f"  • Learning Rate: {config['training'].get('learning_rate', 'N/A')}")
        print(f"  • Loss Strategy: {config['training']['loss'].get('strategy', 'N/A')}")
        print(f"  • Output Directory: {output_dir}")
        print(f"  • Mixed Precision: {config['training'].get('mixed_precision', 'N/A')}")
        print(f"  • Gradient Checkpointing: {config['optimization'].get('gradient_checkpointing', 'N/A')}")
        
        # Create and run trainer
        logger.info("Creating RefactoredBalancedRewardTrainer...")
        trainer = RefactoredBalancedRewardTrainer(config)
        
        # Start training
        logger.info("Starting training...")
        start_time = time.time()
        
        try:
            results = trainer.train()
            training_time = time.time() - start_time
            
            # Add model path to results
            results['model_path'] = output_dir
            
            print_results(True, results, training_time)
            
            logger.info("Training completed successfully")
            return True
            
        except Exception as e:
            training_time = time.time() - start_time
            logger.error(f"Training failed: {e}")
            print_results(False, {"error": str(e)}, training_time)
            return False
            
        finally:
            # Cleanup
            try:
                trainer.cleanup()
                logger.info("Resources cleaned up successfully")
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")
    
    except Exception as e:
        logger.error(f"Script failed: {e}")
        print_results(False, {"error": str(e)}, 0.0)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
