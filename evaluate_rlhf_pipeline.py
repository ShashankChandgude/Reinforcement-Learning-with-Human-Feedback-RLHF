#!/usr/bin/env python3
"""
Comprehensive evaluation script for the RLHF pipeline.
Tests the complete pipeline and generates meaningful results.
"""

import json
import os
from typing import List, Dict, Any
from evaluation.rlhf_evaluator import run_rlhf_evaluation
from data.data_loader import load_dataset
from transformers import AutoTokenizer
from utils.logging_utils import setup_logger


def load_test_data():
    """Load test data for evaluation."""
    logger = setup_logger("evaluation")
    
    # Load a small test set from HH-RLHF
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset_cfg = {
        "loader": "preference",
        "name": "Anthropic/hh-rlhf",
        "subset_size": 100,  # Small test set
        "max_seq_length": 256,
        "clean": True,
        "tokenizer": {
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "pt"
        }
    }
    
    logger.info("Loading test data...")
    dataset = load_dataset(tokenizer, dataset_cfg)
    
    # Extract test data
    test_prompts = []
    test_chosen = []
    test_rejected = []
    
    for i in range(min(20, len(dataset))):  # Use first 20 examples
        example = dataset[i]
        test_prompts.append(example['prompt'])
        test_chosen.append(example['chosen_text'])
        test_rejected.append(example['rejected_text'])
    
    logger.info(f"Loaded {len(test_prompts)} test examples")
    
    return {
        "prompts": test_prompts,
        "chosen_responses": test_chosen,
        "rejected_responses": test_rejected
    }


def evaluate_baseline_vs_trained():
    """Compare baseline model vs trained model."""
    logger = setup_logger("evaluation")
    
    # Load test data
    test_data = load_test_data()
    
    # Check if trained model exists
    trained_model_dir = "models/reward_model_preference"
    if not os.path.exists(trained_model_dir):
        logger.error(f"Trained model not found at {trained_model_dir}")
        return None
    
    logger.info("Evaluating trained model...")
    
    # Evaluate trained model
    results = run_rlhf_evaluation(
        model_dir=trained_model_dir,
        test_prompts=test_data["prompts"],
        test_references=test_data["chosen_responses"],  # Use chosen as reference
        preference_data=test_data
    )
    
    return results


def generate_evaluation_report(results: Dict[str, Any]) -> str:
    """Generate a comprehensive evaluation report."""
    report = []
    report.append("# RLHF Pipeline Evaluation Report")
    report.append("=" * 50)
    report.append("")
    
    # Model Quality Metrics
    if "model_quality" in results:
        report.append("## Model Quality Metrics")
        report.append("")
        metrics = results["model_quality"]
        for metric, value in metrics.items():
            if isinstance(value, float):
                report.append(f"- **{metric}**: {value:.4f}")
            else:
                report.append(f"- **{metric}**: {value}")
        report.append("")
    
    # Preference Alignment
    if "preference_alignment" in results:
        report.append("## Preference Alignment Metrics")
        report.append("")
        pref_metrics = results["preference_alignment"]
        for metric, value in pref_metrics.items():
            if isinstance(value, float):
                report.append(f"- **{metric}**: {value:.4f}")
            else:
                report.append(f"- **{metric}**: {value}")
        report.append("")
    
    # Sample Generations
    if "sample_generations" in results:
        report.append("## Sample Generations")
        report.append("")
        samples = results["sample_generations"]
        for i, (prompt, response) in enumerate(zip(samples["prompts"], samples["responses"])):
            report.append(f"### Example {i+1}")
            report.append(f"**Prompt**: {prompt[:100]}...")
            report.append(f"**Response**: {response[:200]}...")
            report.append("")
    
    # Analysis
    report.append("## Analysis")
    report.append("")
    
    if "preference_alignment" in results:
        pref_acc = results["preference_alignment"].get("preference_accuracy", 0)
        if pref_acc > 0.6:
            report.append("[GOOD] **Good preference alignment** - The reward model shows strong alignment with human preferences.")
        elif pref_acc > 0.4:
            report.append("[WARNING] **Moderate preference alignment** - The reward model shows some alignment but could be improved.")
        else:
            report.append("[POOR] **Poor preference alignment** - The reward model needs significant improvement.")
    
    if "model_quality" in results:
        bleu = results["model_quality"].get("bleu", 0)
        if bleu > 0.3:
            report.append("[GOOD] **Good text quality** - Generated responses show good coherence.")
        elif bleu > 0.1:
            report.append("[WARNING] **Moderate text quality** - Generated responses are somewhat coherent.")
        else:
            report.append("[POOR] **Poor text quality** - Generated responses need improvement.")
    
    report.append("")
    report.append("## Recommendations")
    report.append("")
    report.append("1. **Increase training data** - Use more preference examples for better learning.")
    report.append("2. **Longer training** - Train for more epochs to improve convergence.")
    report.append("3. **Larger model** - Consider using a larger base model (7B+ parameters).")
    report.append("4. **Better evaluation** - Add human evaluation for more reliable metrics.")
    
    return "\n".join(report)


def main():
    """Main evaluation function."""
    logger = setup_logger("evaluation")
    logger.info("Starting RLHF pipeline evaluation...")
    
    try:
        # Run evaluation
        results = evaluate_baseline_vs_trained()
        
        if results is None:
            logger.error("Evaluation failed - no trained model found")
            return False
        
        # Generate report
        report = generate_evaluation_report(results)
        
        # Save results
        os.makedirs("evaluation", exist_ok=True)
        
        with open("evaluation/rlhf_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        with open("evaluation/rlhf_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        
        # Print report
        print("\n" + "="*60)
        print(report)
        print("="*60)
        
        logger.info("Evaluation completed successfully!")
        logger.info("Results saved to evaluation/rlhf_results.json")
        logger.info("Report saved to evaluation/rlhf_report.md")
        
        return True
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
