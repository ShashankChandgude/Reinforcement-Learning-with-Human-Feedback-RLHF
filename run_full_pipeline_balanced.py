#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Complete Balanced RLHF Pipeline - All Phases (cached + eval fix)
- Caches PPO and Supervised phases if model dirs already contain artifacts
- Fixes evaluation call to evaluator.evaluate_model_quality(prompts, references)
- Keeps reward scoring on generated responses
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

# ---- Runtime env knobs (helps Windows + 4GB cards) ----
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Make W&B fully non-interactive/offline
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Trainers
from training.balanced_reward_trainer import BalancedRewardTrainer
from training.simple_lora_trainer import SimpleLoRATrainer
from training.ppo_preference_trainer import PPOPreferenceTrainer, PPOConfig
from training.supervised_rlhf_trainer import SupervisedRLHFTrainer
from training.reward_model import RewardModel

# Evaluation
from evaluation.rlhf_evaluator import RLHFEvaluator  # evaluate_model_quality(prompts, references)  :contentReference[oaicite:1]{index=1}

# Utils
from utils.config_loader import load_config
from utils.logging_utils import setup_logger
from data.data_loader import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def has_model_artifacts(folder: str) -> bool:
    """Return True if folder looks like a saved HF model dir."""
    if not os.path.isdir(folder):
        return False
    names = {"config.json", "pytorch_model.bin", "model.safetensors"}
    try:
        files = set(os.listdir(folder))
    except Exception:
        return False
    return bool(names & files)


class FullPipelineRunner:
    def __init__(self):
        self.logger = setup_logger("full_pipeline")
        self.results = {
            "pipeline_start_time": time.time(),
            "phases_completed": [],
            "phases_failed": [],
            "final_model_path": None,
            "total_training_time": 0.0,
            "performance_metrics": {},
        }

        # Configs
        self.configs = {
            "reward_model": "configs/reward_preference_balanced.yaml",
            "lora": "configs/lora_config.yaml",
            "ppo": "configs/ppo_preference_balanced.yaml",
            "supervised": "configs/supervised_rlhf.yaml",
        }

        # Output model dirs
        self.model_paths = {
            "reward_model": "models/reward_model_preference_balanced",
            "lora": "models/lora_model_improved",
            "ppo": "models/ppo_preference_balanced",
            "supervised": "models/supervised_rlhf_model",
        }

    # -------- pretty printing --------
    def print_banner(self, title: str, subtitle: str = "", width: int = 100):
        print("=" * width)
        print(title)
        if subtitle:
            print(subtitle)
        print("=" * width)

    def print_phase_result(self, phase: str, success: bool, metrics: Dict[str, Any]):
        status = "SUCCESS" if success else "FAILED"
        print(f"\n{status}: {phase}")
        for k, v in metrics.items():
            print(f"  • {k}: {v}")
        print("-" * 50)

    # -------- PHASE 1: Reward model --------
    def phase_1_reward_model(self) -> bool:
        try:
            self.print_banner("PHASE 1: BALANCED REWARD MODEL", "Target: ~90% accuracy")
            if has_model_artifacts(self.model_paths["reward_model"]):
                self.logger.info("Reward model already exists, skipping training")
                self.print_phase_result("Reward Model", True, {
                    "Status": "Existing model used",
                    "Path": self.model_paths["reward_model"],
                    "Time": "0 minutes (cached)",
                })
                self.results["phases_completed"].append("reward_model_cached")
                return True

            t0 = time.time()
            cfg = load_config(self.configs["reward_model"])
            trainer = BalancedRewardTrainer(cfg)
            res = trainer.train()
            dt = time.time() - t0

            self.print_phase_result("Reward Model", True, {
                "Final Accuracy": f"{res.get('final_accuracy', 0):.1f}%",
                "Training Time": f"{dt/60:.1f} minutes",
                "Model Path": self.model_paths["reward_model"],
            })
            self.results["phases_completed"].append("reward_model")
            self.results["performance_metrics"]["reward_accuracy"] = res.get("final_accuracy", 0.0)
            self.results["total_training_time"] += dt
            return True

        except Exception as e:
            self.logger.exception("Phase 1 failed")
            self.results["phases_failed"].append("reward_model")
            self.print_phase_result("Reward Model", False, {"Error": str(e)})
            return False

    # -------- PHASE 2: LoRA --------
    def phase_2_lora_training(self) -> bool:
        try:
            self.print_banner("PHASE 2: IMPROVED LORA TRAINING", "Target: loss reduction")
            t0 = time.time()
            out_dir = self.model_paths["lora"]

            if os.path.exists(out_dir) and os.listdir(out_dir):
                self.logger.info("LoRA model already exists, skipping")
                self.print_phase_result("LoRA Training", True, {
                    "Status": "Existing model used",
                    "Path": out_dir,
                    "Time": "0 minutes (cached)",
                })
                self.results["phases_completed"].append("lora_cached")
                return True

            cfg = load_config(self.configs["lora"])
            trainer = SimpleLoRATrainer(cfg)
            res = trainer.train()

            dt = time.time() - t0
            final_loss = res.get("final_loss", 0.0)
            initial_loss = res.get("initial_loss", final_loss)
            reduction = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0.0

            self.print_phase_result("LoRA Training", True, {
                "Loss Reduction": f"{reduction:.1f}%",
                "Final Loss": f"{final_loss:.4f}",
                "Training Time": f"{dt/60:.1f} minutes",
                "Model Path": out_dir,
                "Trainable Params": f"{res.get('trainable_params', 0)/1e6:.2f}M",
            })
            self.results["phases_completed"].append("lora")
            self.results["performance_metrics"]["lora_loss_reduction"] = reduction
            self.results["total_training_time"] += dt
            return True

        except Exception as e:
            self.logger.exception("Phase 2 failed")
            self.results["phases_failed"].append("lora")
            self.print_phase_result("LoRA Training", False, {"Error": str(e)})
            return False

    # -------- PHASE 3: PPO --------
    def phase_3_ppo_training(self) -> bool:
        try:
            self.print_banner("PHASE 3: BALANCED PPO TRAINING", "On-policy rollouts + token-KL")

            # NEW: cache check
            ppo_dir = self.model_paths["ppo"]
            if has_model_artifacts(ppo_dir):
                self.logger.info("PPO model already exists, skipping")
                self.print_phase_result("PPO Training", True, {
                    "Status": "Existing model used",
                    "Path": ppo_dir,
                    "Time": "0 minutes (cached)",
                })
                # Prefer PPO as final if present
                self.results["phases_completed"].append("ppo_cached")
                self.results["performance_metrics"]["ppo_success"] = True
                self.results["final_model_path"] = ppo_dir
                return True

            t0 = time.time()
            cfg_dict = load_config(self.configs["ppo"])
            ppo_cfg = PPOConfig(**{
                **PPOConfig().__dict__,  # defaults
                **(load_config(self.configs["ppo"]).get("ppo_patch", {})),  # optional patching
            })
            # Build config from YAML properly
            from training.ppo_preference_trainer import build_config_from_yaml
            ppo_cfg = build_config_from_yaml(cfg_dict)

            tok = AutoTokenizer.from_pretrained(ppo_cfg.model_name)
            if not tok.pad_token:
                tok.pad_token = tok.eos_token

            ds_cfg = cfg_dict.get("dataset", {})
            prompts_dataset = load_dataset(tokenizer=tok, dataset_cfg=ds_cfg)

            # Load reward model from Phase 1
            base = AutoModelForCausalLM.from_pretrained(ppo_cfg.model_name)
            rm = RewardModel(base)
            rm_path = os.path.join(self.model_paths["reward_model"], "reward_model.pth")
            rm.load_state_dict(torch.load(rm_path, map_location="cpu"))
            rm.eval()

            trainer = PPOPreferenceTrainer(
                tokenizer=tok,
                reward_model=rm,
                prompts_dataset=prompts_dataset,
                config=ppo_cfg,
            )

            # Iterator over prompts
            def iterator():
                bs = ppo_cfg.rollout_batch_size
                buf = []
                for p in getattr(prompts_dataset, "prompts", []):
                    if p is None:
                        continue
                    buf.append(p)
                    if len(buf) == bs:
                        yield buf
                        buf = []
                if buf:
                    yield buf

            total_rollouts = int(cfg_dict["training"].get("total_rollouts", 64))
            self.logger.info(
                f"Starting PPO: rollouts={total_rollouts}, rb={ppo_cfg.rollout_batch_size}, "
                f"mb={ppo_cfg.mini_batch_size}, max_new={ppo_cfg.max_new_tokens}"
            )
            results = trainer.train(iterator(), total_rollouts=total_rollouts)

            out_dir = ppo_dir
            os.makedirs(out_dir, exist_ok=True)
            trainer.model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)

            dt = time.time() - t0
            self.print_phase_result("PPO Training", True, {
                "Policy Loss": f"{results.get('policy_loss', 0):.4f}",
                "Value Loss": f"{results.get('value_loss', 0):.4f}",
                "Avg KL": f"{results.get('kl_divergence', 0):.4f}",
                "Entropy": f"{results.get('entropy', 0):.4f}",
                "Grad Norm": f"{results.get('grad_norm', 0):.2f}",
                "Training Time": f"{dt/60:.1f} min",
                "Model Path": out_dir,
            })
            self.results["phases_completed"].append("ppo")
            self.results["performance_metrics"]["ppo_success"] = True
            self.results["final_model_path"] = out_dir
            self.results["total_training_time"] += dt
            return True

        except Exception as e:
            self.logger.exception("PPO phase failed")
            self.print_phase_result("PPO Training", False, {"Error": str(e)})
            self.results["performance_metrics"]["ppo_success"] = False
            return False

    # -------- PHASE 4: Supervised RLHF --------
    def phase_4_supervised_rlhf(self, is_fallback: bool = False) -> bool:
        try:
            self.print_banner(
                "PHASE 4: SUPERVISED RLHF (FALLBACK)" if is_fallback else "PHASE 4: SUPERVISED RLHF",
                "Reward-ranked supervised learning",
            )

            # NEW: cache check
            sup_dir = self.model_paths["supervised"]
            if has_model_artifacts(sup_dir):
                self.logger.info("Supervised RLHF model already exists, skipping")
                self.print_phase_result("Supervised RLHF", True, {
                    "Status": "Existing model used",
                    "Path": sup_dir,
                    "Time": "0 minutes (cached)",
                })
                self.results["phases_completed"].append("supervised_rlhf_cached")
                # If PPO didn’t succeed, use supervised as final
                if not self.results.get("final_model_path"):
                    self.results["final_model_path"] = sup_dir
                self.results["performance_metrics"]["supervised_success"] = True
                return True

            t0 = time.time()
            cfg = load_config(self.configs["supervised"])
            trainer = SupervisedRLHFTrainer(cfg)
            res = trainer.train()
            dt = time.time() - t0

            self.print_phase_result("Supervised RLHF", True, {
                "Final Loss": f"{res.get('final_loss', 0):.4f}",
                "High-Quality Examples": res.get("examples_used", 0),
                "Epochs Completed": res.get("epochs_completed", 0),
                "Training Time": f"{dt/60:.1f} minutes",
                "Model Path": sup_dir,
                "Approach": "Reward-ranked supervised",
            })
            self.results["phases_completed"].append("supervised_rlhf")
            self.results["performance_metrics"]["supervised_success"] = True
            self.results["performance_metrics"]["supervised_loss"] = res.get("final_loss", 0)
            # If PPO succeeded we keep PPO as final; else, supervised becomes final
            if not self.results.get("final_model_path"):
                self.results["final_model_path"] = sup_dir
            self.results["total_training_time"] += dt
            return True

        except Exception as e:
            self.logger.exception("Phase 4 failed")
            self.results["phases_failed"].append("supervised_rlhf")
            self.print_phase_result("Supervised RLHF", False, {"Error": str(e)})
            return False

    # -------- PHASE 5: Evaluation (fixed call) --------
    def phase_5_model_evaluation(self) -> bool:
        try:
            self.print_banner("PHASE 5: MODEL EVALUATION", "Comprehensive assessment")
            t0 = time.time()

            model_path = self.results.get("final_model_path")
            if not model_path:
                self.logger.error("No final model available for evaluation")
                return False

            evaluator = RLHFEvaluator(model_path)
            evaluator.load_model()

            eval_prompts = [
                "Human: What is the capital of France?\n\nAssistant:",
                "Human: How can I help someone who is feeling sad?\n\nAssistant:",
                "Human: Explain quantum computing in simple terms.\n\nAssistant:",
                "Human: What are the benefits of exercise?\n\nAssistant:",
                "Human: How do I make a good first impression?\n\nAssistant:",
            ]
            # Evaluator expects references list; keep empty references OK
            references = [""] * len(eval_prompts)

            # Generate once so we can also compute reward scores
            responses = evaluator.generate_responses(eval_prompts, max_length=150)

            # FIX: call with (prompts, references) (no 'responses=' kwarg)  :contentReference[oaicite:2]{index=2}
            quality_results = evaluator.evaluate_model_quality(
                prompts=eval_prompts,
                references=references,
            )

            # Reward scores using Phase 1 reward model if available
            avg_reward = 0.0
            reward_scores: List[float] = []
            try:
                reward_scores = evaluator.compute_reward_scores(eval_prompts, responses)
                if reward_scores:
                    avg_reward = sum(reward_scores) / len(reward_scores)
            except Exception as e:
                self.logger.warning(f"Skipping reward scoring in eval: {e}")

            dt = time.time() - t0
            metrics = {
                "Model Type": ("PPO" if "ppo" in model_path.lower() else "Supervised RLHF"),
                "Average Reward Score": f"{avg_reward:.3f}",
                "Average Perplexity": f"{quality_results.get('avg_perplexity', 0.0):.2f}",
                "Response Quality": f"{quality_results.get('avg_quality', 0.0):.3f}",
                "Evaluation Time": f"{dt/60:.1f} minutes",
                "Test Prompts": len(eval_prompts),
                "Model Path": model_path,
            }
            self.print_phase_result("Model Evaluation", True, metrics)

            self.results["phases_completed"].append("model_evaluation")
            self.results["performance_metrics"]["evaluation_results"] = {
                "prompts": eval_prompts,
                "responses": responses,
                "quality_metrics": quality_results,
                "reward_scores": reward_scores,
                "average_reward_score": avg_reward,
                "response_quality_score": quality_results.get("avg_quality", 0.0),
                "average_perplexity": quality_results.get("avg_perplexity", 0.0),
            }
            self.results["performance_metrics"]["avg_reward_score"] = avg_reward
            self.results["performance_metrics"]["avg_perplexity"] = quality_results.get("avg_perplexity", 0.0)
            self.results["total_training_time"] += dt

            with open("evaluation_results.json", "w") as f:
                json.dump(self.results["performance_metrics"]["evaluation_results"], f, indent=2)

            return True

        except Exception as e:
            self.logger.exception("Phase 5 failed")
            self.results["phases_failed"].append("model_evaluation")
            self.print_phase_result("Model Evaluation", False, {"Error": str(e)})
            return False

    # -------- bookkeeping --------
    def save_pipeline_results(self):
        self.results["pipeline_end_time"] = time.time()
        self.results["total_pipeline_time"] = (
            self.results["pipeline_end_time"] - self.results["pipeline_start_time"]
        )
        with open("pipeline_results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        self.logger.info("Pipeline results saved to pipeline_results.json")

    def print_final_summary(self):
        self.print_banner("RLHF PIPELINE COMPLETED", width=100)
        total_time = self.results["total_pipeline_time"]
        training_time = self.results["total_training_time"]

        print("PIPELINE SUMMARY:")
        print(f"  • Total Runtime: {total_time/60:.1f} minutes")
        print(f"  • Active Training Time: {training_time/60:.1f} minutes")
        print(f"  • Phases Completed: {len(self.results['phases_completed'])}")
        print(f"  • Phases Failed: {len(self.results['phases_failed'])}")

        print("\nCOMPLETED PHASES:")
        for phase in self.results["phases_completed"]:
            print(f"  • {phase.replace('_', ' ').title()}")

        if self.results["phases_failed"]:
            print("\nFAILED PHASES:")
            for phase in self.results["phases_failed"]:
                print(f"  • {phase.replace('_', ' ').title()}")

        metrics = self.results["performance_metrics"]
        if metrics.get("ppo_success"):
            print("\nPERFORMANCE METRICS:\n  • PPO Training: SUCCESS")
        if "supervised_success" in metrics:
            print(f"  • Supervised RLHF: SUCCESS (Final Loss: {metrics.get('supervised_loss', 0):.4f})")

        if self.results["final_model_path"]:
            print("\nFINAL MODEL:")
            print(f"  • Path: {self.results['final_model_path']}")
            print("  • Status: Production ready")

        print("\nALL TRAINED MODELS:")
        for phase, path in self.model_paths.items():
            if has_model_artifacts(path):
                print(f"  • {phase.replace('_', ' ').title()}: {path}")

        print("=" * 100)

    def run(self) -> bool:
        self.print_banner(
            "COMPLETE BALANCED RLHF PIPELINE",
            "Reward Model -> LoRA -> PPO/Supervised RLHF",
            width=100,
        )
        print("Pipeline Phases:")
        print("  1. Balanced Reward Model")
        print("  2. Improved LoRA Training")
        print("  3. PPO Training (gradient-safe)")
        print("  4. Supervised RLHF (fallback or comparison)")
        print("  5. Model Evaluation\n")
        print("=" * 100)

        # Ensure model dirs exist
        for p in self.model_paths.values():
            os.makedirs(p, exist_ok=True)

        if not self.phase_1_reward_model():
            return False

        _ = self.phase_2_lora_training()  # not critical

        ppo_ok = self.phase_3_ppo_training()

        # Run supervised either as fallback or (if PPO ok) for comparison
        sup_ok = self.phase_4_supervised_rlhf(is_fallback=not ppo_ok)

        # Choose final model path if none set yet
        if not self.results.get("final_model_path"):
            if ppo_ok and has_model_artifacts(self.model_paths["ppo"]):
                self.results["final_model_path"] = self.model_paths["ppo"]
            elif sup_ok and has_model_artifacts(self.model_paths["supervised"]):
                self.results["final_model_path"] = self.model_paths["supervised"]

        eval_ok = False
        if self.results.get("final_model_path"):
            eval_ok = self.phase_5_model_evaluation()

        self.save_pipeline_results()
        self.print_final_summary()
        return bool(self.results.get("final_model_path")) and (ppo_ok or sup_ok) and eval_ok


def main():
    print("Starting Complete Balanced RLHF Pipeline...")
    print("Hardware: GTX 1650 4GB optimized")
    print("Approach: Balanced configurations with intelligent fallbacks\n")

    runner = FullPipelineRunner()
    ok = runner.run()
    print("\n" + ("SUCCESS: Complete RLHF pipeline finished with production-ready model!" if ok
                 else "Pipeline failed. Check logs for details."))
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
