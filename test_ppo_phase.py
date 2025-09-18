# test_ppo_phase.py
import os, sys, time, traceback
import faulthandler
faulthandler.enable()

print(f">>> starting test_ppo_phase.py", flush=True)
print(f"__file__: {__file__}", flush=True)
print(f"cwd     : {os.getcwd()}", flush=True)

# Show runtime env early
try:
    import platform
    import torch
    print(f"python  : {sys.version.split()[0]}", flush=True)
    print(f"torch   : {getattr(torch, '__version__', 'n/a')}", flush=True)
    print(f"cuda    : {torch.cuda.is_available()}", flush=True)
except Exception as _e:
    print(f"[warn] could not import torch early: {_e}", flush=True)

# Imports after the early banner
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.config_loader import load_config
from training.Reward_Model.reward_model import RewardModel
from data.data_loader import load_dataset
from training.PPO_trainer.ppo_preference_trainer import (
    PPOPreferenceTrainer,
    build_config_from_yaml,
    prompt_loader_from_dataset,
)

def banner(title, subtitle=""):
    print("=" * 70, flush=True)
    print(f"🧪 {title}", flush=True)
    if subtitle:
        print(subtitle, flush=True)
    print("=" * 70, flush=True)

def main():
    print("🚀 Testing PPO Phase in Isolation...", flush=True)
    print("This verifies on-policy rollouts, token-level KL, and prompt-state value.\n", flush=True)
    banner("PPO PHASE ISOLATION TEST", "Stable PPO sanity check")

    # Config path
    ppo_cfg_path = os.path.join("configs", "ppo_preference_balanced.yaml")
    print(f"Config path: {ppo_cfg_path}", flush=True)
    print(f"Config exists: {os.path.exists(ppo_cfg_path)}", flush=True)
    if not os.path.exists(ppo_cfg_path):
        raise FileNotFoundError(f"Config not found: {ppo_cfg_path}")

    # Load YAML and map to trainer config
    raw_cfg = load_config(ppo_cfg_path)
    print(f"Loaded YAML keys: {list(raw_cfg.keys())}", flush=True)
    if "model" not in raw_cfg:
        raise KeyError("`model` key missing in PPO YAML.")
    if "training" not in raw_cfg:
        raise KeyError("`training` section missing in PPO YAML.")
    if "gen" not in raw_cfg:
        raw_cfg["gen"] = {}

    ppo_cfg = build_config_from_yaml(raw_cfg)
    print(f"Model: {raw_cfg['model']}", flush=True)
    print(f"Rollout batch size: {ppo_cfg.rollout_batch_size} | Mini-batch: {ppo_cfg.mini_batch_size} | "
          f"Total rollouts: {ppo_cfg.total_rollouts}", flush=True)

    # Tokenizer
    model_name = raw_cfg["model"]
    tok = AutoTokenizer.from_pretrained(model_name)
    if not tok.pad_token:
        tok.pad_token = tok.eos_token
    print("Tokenizer loaded.", flush=True)

    # Dataset -> prompts for rollouts
    print("Loading dataset...", flush=True)
    ds = load_dataset(tokenizer=tok, dataset_cfg=raw_cfg.get("dataset", {}))
    if hasattr(ds, "prompts"):
        print(f"Dataset ready. prompts={len(ds.prompts)}", flush=True)
    else:
        raise RuntimeError("Dataset has no `.prompts`. Check dataset loader/prep.")

    # Reward model + weights (use YAML path)
    reward_dir = raw_cfg.get("reward_model_dir", os.path.join("models", "reward_model_preference_balanced"))
    weights = os.path.join(reward_dir, "reward_model.pth")
    print(f"Reward weights: {weights} (exists={os.path.exists(weights)})", flush=True)
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Reward model weights not found at {weights}")

    rm_base = AutoModelForCausalLM.from_pretrained(model_name)
    rm = RewardModel(rm_base)
    # Safe load of a state_dict
    try:
        state = torch.load(weights, map_location="cpu", weights_only=True)  # PyTorch 2.5+
    except TypeError:
        # Fallback for older PyTorch
        state = torch.load(weights, map_location="cpu")
    rm.load_state_dict(state)
    rm.eval()
    print("Reward model loaded.", flush=True)

    # Trainer
    trainer = PPOPreferenceTrainer(
        tokenizer=tok,
        reward_model=rm,
        prompts_dataset=ds,
        config=ppo_cfg,
    )
    print("PPO trainer constructed.", flush=True)

    # Prompt iterator matching rollout_batch_size
    iterator = prompt_loader_from_dataset(ds, ppo_cfg.rollout_batch_size)

    # Preflight snapshot
    print("\n🔎 Preflight: collecting one rollout batch...", flush=True)
    pre_buf = trainer.collect_rollouts(iterator, cap=ppo_cfg.rollout_batch_size)
    n = pre_buf["full_ids"].size(0)
    print(f"Preflight batch size: {n}", flush=True)
    if n > 0:
        kls = pre_buf["kls"]; rews = pre_buf["rewards"]; vals = pre_buf["values"]
        print(f"KL mean {kls.mean().item():.4f} std {kls.std().item():.4f}", flush=True)
        print(f"R  mean {rews.mean().item():.4f} std {rews.std().item():.4f}", flush=True)
        print(f"V  mean {vals.mean().item():.4f} std {vals.std().item():.4f}", flush=True)
    else:
        print("No samples in preflight.", flush=True)

    # Fresh iterator for training (since we consumed one batch above)
    iterator = prompt_loader_from_dataset(ds, ppo_cfg.rollout_batch_size)

    # Train
    print("\n🔄 Starting PPO training...", flush=True)
    t0 = time.time()
    res = trainer.train(iterator, total_rollouts=ppo_cfg.total_rollouts)
    dt = time.time() - t0

    print("\n" + "=" * 70, flush=True)
    if res.get("steps", 0) > 0:
        print("✅ PPO isolation run completed.", flush=True)
        print(f"Avg Policy Loss : {res.get('policy_loss', 0):.4f}", flush=True)
        print(f"Avg Value Loss  : {res.get('value_loss', 0):.4f}", flush=True)
        print(f"Avg KL          : {res.get('kl_divergence', 0):.4f}", flush=True)
        print(f"Avg Entropy     : {res.get('entropy', 0):.4f}", flush=True)
        print(f"Avg Grad Norm   : {res.get('grad_norm', 0):.4f}", flush=True)
        print(f"Took            : {dt:.1f}s", flush=True)
        if getattr(trainer, "training_history", None):
            tail = trainer.training_history[-3:] if len(trainer.training_history) >= 3 else trainer.training_history
            print("\nRecent steps:", flush=True)
            for m in tail:
                print(f"  step={m['step']:>5}  pol={m['policy_loss']:.4f}  "
                      f"val={m['value_loss']:.4f}  tot={m['total_loss']:.4f}  "
                      f"kl={m['kl_divergence']:.4f}  ent={m['entropy']:.4f}  "
                      f"gn={m['grad_norm']:.4f}  beta_kl={m['beta_kl']:.4f}", flush=True)
    else:
        print("❌ PPO isolation produced no steps. Check iterator/config.", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print("\n[FATAL] Unhandled exception in test_ppo_phase.py:", flush=True)
        traceback.print_exc()
        sys.exit(1)
