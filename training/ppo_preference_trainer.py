# training/ppo_preference_trainer.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Iterable
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


# ============================= Config =============================

@dataclass
class PPOConfig:
    # model
    model_name: str = "EleutherAI/gpt-neo-125M"

    # generation (prefer YAML.gen; fall back to YAML.training*)
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    do_sample: bool = True

    # rollout
    rollout_batch_size: int = 8
    total_rollouts: int = 64

    # PPO update
    ppo_epochs: int = 2
    mini_batch_size: int = 4
    clip_epsilon: float = 0.1
    entropy_coeff: float = 0.005
    value_coeff: float = 0.5
    max_grad_norm: float = 1.0
    policy_learning_rate: float = 5e-6
    value_learning_rate: float = 1e-5
    gradient_accumulation_steps: int = 1

    # KL control
    beta_kl: float = 0.02
    kl_target: float = 0.03
    adaptive_kl: bool = True

    # advantage/value
    normalize_advantages: bool = True
    advantage_clip: float = 5.0
    value_clip_epsilon: float = 0.2

    # value warmup
    pretrain_value_steps: int = 10
    warmup_steps: int = 20  # placeholder for schedulers if you add them

    # runtime / memory
    logging_steps: int = 10
    mixed_precision: bool = False  # fp16 autocast on CUDA
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # placement flags (YAML: optimization)
    policy_on_gpu: bool = True
    reward_model_on_cpu: bool = True
    reference_policy_on_cpu: bool = True
    gradient_checkpointing: bool = False

    # dataset prompt cap (from YAML.dataset.max_seq_length)
    max_prompt_tokens: int = 128

    # hygiene
    clear_cache_between_epochs: bool = True


def build_config_from_yaml(yaml_cfg: Dict[str, Any]) -> PPOConfig:
    """Map YAML dict into PPOConfig. Supports both YAML.gen and YAML.training for gen keys."""
    tr = yaml_cfg.get("training", {}) or {}
    gen = yaml_cfg.get("gen", {}) or {}
    ds  = yaml_cfg.get("dataset", {}) or {}
    opt = yaml_cfg.get("optimization", {}) or {}

    # If gen.* is missing, fall back to training.*
    g_max_new_tokens = gen.get("max_new_tokens", tr.get("max_new_tokens", 64))
    g_temperature    = gen.get("temperature",    tr.get("temperature", 0.7))
    g_top_p          = gen.get("top_p",          tr.get("top_p", 0.9))
    g_top_k          = gen.get("top_k",          tr.get("top_k", 0))
    g_do_sample      = gen.get("do_sample",      tr.get("do_sample", True))

    return PPOConfig(
        model_name=yaml_cfg.get("model", "EleutherAI/gpt-neo-125M"),
        # generation
        max_new_tokens=int(g_max_new_tokens),
        temperature=float(g_temperature),
        top_p=float(g_top_p),
        top_k=int(g_top_k),
        do_sample=bool(g_do_sample),
        # rollout
        rollout_batch_size=int(tr.get("rollout_batch_size", 8)),
        total_rollouts=int(tr.get("total_rollouts", 64)),
        # PPO
        ppo_epochs=int(tr.get("ppo_epochs", 2)),
        mini_batch_size=int(tr.get("mini_batch_size", 4)),
        clip_epsilon=float(tr.get("clip_epsilon", 0.1)),
        entropy_coeff=float(tr.get("entropy_coeff", 0.005)),
        value_coeff=float(tr.get("value_coeff", 0.5)),
        max_grad_norm=float(tr.get("max_grad_norm", 1.0)),
        policy_learning_rate=float(tr.get("policy_learning_rate", tr.get("policy_lr", 5e-6))),
        value_learning_rate=float(tr.get("value_learning_rate", tr.get("value_lr", 1e-5))),
        gradient_accumulation_steps=int(tr.get("gradient_accumulation_steps", 1)),
        # KL
        beta_kl=float(tr.get("beta_kl", 0.02)),
        kl_target=float(tr.get("kl_target", 0.03)),
        adaptive_kl=bool(tr.get("adaptive_kl", True)),
        # advantages / values
        normalize_advantages=bool(tr.get("normalize_advantages", True)),
        advantage_clip=float(tr.get("advantage_clip", 5.0)),
        value_clip_epsilon=float(tr.get("value_clip_epsilon", 0.2)),
        # warmup
        pretrain_value_steps=int(tr.get("pretrain_value_steps", 10)),
        warmup_steps=int(tr.get("warmup_steps", 20)),
        # misc
        logging_steps=int(tr.get("logging_steps", 10)),
        mixed_precision=bool(tr.get("mixed_precision", False)),
        # optimization
        policy_on_gpu=bool(opt.get("policy_on_gpu", True)),
        reward_model_on_cpu=bool(opt.get("reward_model_on_cpu", True)),
        reference_policy_on_cpu=bool(opt.get("reference_policy_on_cpu", True)),
        gradient_checkpointing=bool(opt.get("gradient_checkpointing", False)),
        clear_cache_between_epochs=bool(opt.get("clear_cache_between_epochs", True)),
        # dataset cap
        max_prompt_tokens=int(ds.get("max_seq_length", 128)),
    )


# ============================= Trainer =============================

class PPOPreferenceTrainer:
    """
    PPO for RLHF (bandit-style):
      • On-policy rollouts from current policy
      • Token-level KL(p‖q) vs frozen reference policy (non-negative)
      • Reward from external reward model on full sequence
      • Value at prompt state (last prompt token), no lm_head to save VRAM
      • Response-only losses with EOS-aware slicing
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        reward_model: nn.Module,
        prompts_dataset: Any,  # .prompts or iterable of dicts with "prompt"
        config: PPOConfig,
    ):
        self.cfg = config
        self.tokenizer = tokenizer

        # Devices
        self.policy_device = "cuda" if (self.cfg.policy_on_gpu and torch.cuda.is_available()) else "cpu"
        self.ref_device = "cpu" if self.cfg.reference_policy_on_cpu else self.policy_device
        self.rm_device = "cpu" if self.cfg.reward_model_on_cpu else self.policy_device

        # Policy (trainable)
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_name)
        self.model.config.use_cache = False  # training + memory
        self.model.to(self.policy_device)

        # Gradient checkpointing (disabled by default on 4GB)
        if self.cfg.gradient_checkpointing:
            try:
                self.model.gradient_checkpointing_enable()
            except Exception:
                pass

        # Reference (frozen)
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.cfg.model_name)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.ref_model.to(self.ref_device)

        # Value head on top of hidden states
        hidden_size = self.model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1).to(self.policy_device)

        # Optimizers
        self.policy_optim = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.policy_learning_rate)
        self.value_optim = torch.optim.AdamW(self.value_head.parameters(), lr=self.cfg.value_learning_rate)

        # Reward model (frozen)
        self.reward_model = reward_model.eval()
        for p in self.reward_model.parameters():
            p.requires_grad = False
        self.reward_model.to(self.rm_device)

        # Data
        self.prompts_dataset = prompts_dataset

        # Generation config (force use_cache=False to curb KV memory)
        self.gen_cfg = GenerationConfig(
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=self.cfg.do_sample,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
            pad_token_id=(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id),
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=False,
        )

        # AMP (optional)
        self._amp_dtype = torch.float16 if (self.cfg.mixed_precision and torch.cuda.is_available()) else None
        self._amp_enabled = self._amp_dtype is not None

        self.global_step = 0
        self.training_history: List[Dict] = []

    # ------------------------ Tokenization / Generation ------------------------

    def _batch_tokenize_prompts(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_prompt_tokens,  # enforce prompt cap
        )
        ids = enc["input_ids"].to(self.policy_device)
        attn = enc["attention_mask"].to(self.policy_device)
        lens = attn.sum(dim=1).tolist()
        return ids, attn, lens

    @torch.no_grad()
    def _generate(self, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor) -> torch.Tensor:
        # Seatbelt: force at least a few response tokens to avoid empty slices
        min_new = max(4, int(self.cfg.max_new_tokens // 8))
        out = self.model.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            generation_config=self.gen_cfg,
            do_sample=True,  # enforce sampling
            temperature=max(0.7, float(self.cfg.temperature)),
            min_new_tokens=min_new,
        )
        return out

    # ------------------------ EOS-aware response slicing ------------------------

    def _slice_resp_from_logits(self, logits: torch.Tensor, tokens: torch.Tensor, prompt_len: int):
        """
        Align logits (for x[:, :-1]) with targets (x[:, 1:]) and slice response:
           start at prompt_len - 1 in logits, and prompt_len in tokens.
        Returns (resp_logits, resp_tokens) with equal time length, trimmed at first EOS per sample.
        """
        start_log = max(prompt_len - 1, 0)
        resp_logits = logits[:, start_log:, :]   # [B, L_log, V]
        resp_tokens = tokens[:, prompt_len:]     # [B, L_tgt]
        L = min(resp_logits.size(1), resp_tokens.size(1))
        if L <= 0:
            return None, None
        resp_logits = resp_logits[:, :L, :]
        resp_tokens = resp_tokens[:, :L]

        # Trim at first EOS in each sample (optional but stabilizing)
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None and resp_tokens.numel() > 0:
            keep = []
            for i in range(resp_tokens.size(0)):
                row = resp_tokens[i]
                idx = (row == eos_id).nonzero(as_tuple=False)
                cut = int(idx[0].item()) if idx.numel() > 0 else row.size(0)
                keep.append(cut)
            trimmed_tokens = []
            trimmed_logits = []
            for i in range(resp_tokens.size(0)):
                cut = keep[i]
                if cut == 0:
                    continue
                trimmed_tokens.append(resp_tokens[i, :cut])
                trimmed_logits.append(resp_logits[i, :cut, :])
            if not trimmed_tokens:
                return None, None
            resp_tokens = torch.stack(trimmed_tokens, dim=0)
            resp_logits = torch.stack(trimmed_logits, dim=0)

        return resp_logits, resp_tokens

    # ------------------------ Logprob / KL / Value / Reward ------------------------

    @torch.no_grad()
    def _resp_logprob_mean(self, full_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
        """Mean per-token log-prob of the sampled response under current policy."""
        x = full_ids.to(self.policy_device)
        with torch.autocast(device_type="cuda", dtype=self._amp_dtype) if self._amp_enabled else _nullctx():
            logits = self.model(x[:, :-1]).logits  # [B, T-1, V]
        out = self._slice_resp_from_logits(logits, x, prompt_len)
        if out is None:
            return torch.zeros(x.size(0), device=self.policy_device)
        resp_logits, resp_tokens = out
        if resp_logits is None or resp_tokens is None or resp_tokens.size(1) == 0:
            return torch.zeros(x.size(0), device=self.policy_device)
        logp = torch.log_softmax(resp_logits, dim=-1)
        tok_lp = torch.gather(logp, -1, resp_tokens.unsqueeze(-1)).squeeze(-1)  # [B, L]
        return tok_lp.mean(dim=-1)  # [B]

    @torch.no_grad()
    def _resp_token_kl_mean(self, full_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
        """Token-level KL(p‖q) averaged over response tokens (non-negative)."""
        x_p = full_ids.to(self.policy_device)
        x_q = full_ids.to(self.ref_device)
        with torch.autocast(device_type="cuda", dtype=self._amp_dtype) if self._amp_enabled else _nullctx():
            logits_p = self.model(x_p[:, :-1]).logits
        logits_q = self.ref_model(x_q[:, :-1]).logits  # ref on CPU by default
        out_p = self._slice_resp_from_logits(logits_p, x_p, prompt_len)
        out_q = self._slice_resp_from_logits(logits_q, x_q, prompt_len)
        if (out_p is None) or (out_q is None):
            return torch.zeros(x_p.size(0), device=self.policy_device)
        resp_logits_p, _ = out_p
        resp_logits_q, _ = out_q
        if (resp_logits_p is None) or (resp_logits_q is None) or (resp_logits_p.size(1) == 0):
            return torch.zeros(x_p.size(0), device=self.policy_device)
        resp_logits_q = resp_logits_q.to(self.policy_device)

        logp = torch.log_softmax(resp_logits_p, dim=-1)
        logq = torch.log_softmax(resp_logits_q, dim=-1)
        p = torch.exp(logp)
        kl_tok = (p * (logp - logq)).sum(dim=-1)  # [B, L]
        return kl_tok.mean(dim=-1)                # [B]

    def _prompt_value(self, full_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
        """Value at the last prompt token state; avoid lm_head to save VRAM."""
        # Ensure inputs are regular autograd tensors (not 'inference' tensors)
        full_ids = full_ids.clone()

        pad = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        attn = (full_ids != pad).long().to(self.policy_device)

        # Call backbone directly if available to skip vocab projection
        backbone = getattr(self.model, "transformer", None) or getattr(self.model, "model", None)
        with torch.autocast(device_type="cuda", dtype=self._amp_dtype) if self._amp_enabled else _nullctx():
            if backbone is not None:
                outputs = backbone(input_ids=full_ids.to(self.policy_device),
                                   attention_mask=attn,
                                   output_hidden_states=True,
                                   use_cache=False)
                last_h = getattr(outputs, "last_hidden_state", None)
                if last_h is None:
                    last_h = outputs.hidden_states[-1]
            else:
                outputs = self.model(full_ids.to(self.policy_device),
                                     attention_mask=attn,
                                     output_hidden_states=True,
                                     use_cache=False)
                last_h = outputs.hidden_states[-1]

        idx = max(prompt_len - 1, 0)
        state = last_h[:, idx, :]  # [B, H]
        return self.value_head(state).squeeze(-1)  # [B]

    @torch.no_grad()
    def _sequence_reward(self, full_ids: torch.Tensor) -> torch.Tensor:
        """Reward from RewardModel over the full sequence."""
        pad = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        attn = (full_ids != pad).long().to(self.rm_device)
        out = self.reward_model(input_ids=full_ids.to(self.rm_device), attention_mask=attn)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.squeeze(-1).to(self.policy_device).float()  # [B]

    # ------------------------ Rollouts ------------------------

    def _collect_one_batch(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """Collect one on-policy rollout batch for PPO."""
        ids, mask, lens = self._batch_tokenize_prompts(prompts)

        # generation can be heavy; keep inference_mode and free cache
        with torch.inference_mode():
            gen_ids = self._generate(ids, mask)  # [B, T]
        # Turn 'inference tensors' into normal autograd-friendly tensors
        gen_ids = gen_ids.clone().detach()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        B = gen_ids.size(0)
        old_logprobs, kls, values, rewards = [], [], [], []
        kept_full_ids, kept_lens = [], []

        for i in range(B):
            pl = int(lens[i])
            # quick response-length heuristic; ensure positive length
            resp_len_est = max(0, gen_ids.size(1) - (pl + 1))
            if resp_len_est <= 0:
                continue

            seq = gen_ids[i : i + 1]  # [1, T]

            lp = self._resp_logprob_mean(seq, pl)     # [1]
            kl = self._resp_token_kl_mean(seq, pl)    # [1]

            # rollout-time value doesn't need grads; avoid "inference tensor" autograd error
            with torch.no_grad():
                v = self._prompt_value(seq, pl)       # [1]
            v = v.detach()

            r  = self._sequence_reward(seq)           # [1]

            # If slicing failed inside helpers (returned zeros) we still keep the sample;
            # the guards prevent None -> log_softmax(None) crashes.

            kept_full_ids.append(seq.squeeze(0))
            kept_lens.append(pl)
            old_logprobs.append(lp.squeeze(0).to("cpu"))
            kls.append(kl.squeeze(0).to("cpu"))
            values.append(v.squeeze(0).to("cpu"))
            rewards.append(r.squeeze(0).to("cpu"))

        # If nothing valid, return empty tensors
        if len(kept_full_ids) == 0:
            return {
                "full_ids": torch.empty(0, 1, dtype=torch.long, device=self.policy_device),
                "prompt_lens": torch.empty(0, dtype=torch.long, device=self.policy_device),
                "old_logprobs": torch.empty(0, device=self.policy_device),
                "kls": torch.empty(0, device=self.policy_device),
                "values": torch.empty(0, device=self.policy_device),
                "rewards": torch.empty(0, device=self.policy_device),
            }

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "full_ids": torch.stack(kept_full_ids, dim=0).to(self.policy_device),    # [Bv, T]
            "prompt_lens": torch.tensor(kept_lens, device=self.policy_device, dtype=torch.long),
            "old_logprobs": torch.stack(old_logprobs, dim=0).to(self.policy_device),  # [Bv]
            "kls": torch.stack(kls, dim=0).to(self.policy_device),                    # [Bv]
            "values": torch.stack(values, dim=0).to(self.policy_device),              # [Bv]
            "rewards": torch.stack(rewards, dim=0).to(self.policy_device),            # [Bv]
        }

    def collect_rollouts(self, iterator: Iterable[List[str]], cap: int) -> Dict[str, torch.Tensor]:
        """Accumulate rollout batches until we reach `cap` samples."""
        bufs = []
        n = 0
        for prompts in iterator:
            buf = self._collect_one_batch(prompts)
            if buf["full_ids"].numel() == 0:
                # skip empty collection
                continue
            bufs.append(buf)
            n += buf["full_ids"].size(0)
            if n >= cap:
                break
        if not bufs:
            # return a fully empty buffer
            device = self.policy_device
            return {
                "full_ids": torch.empty(0, 1, dtype=torch.long, device=device),
                "prompt_lens": torch.empty(0, dtype=torch.long, device=device),
                "old_logprobs": torch.empty(0, device=device),
                "kls": torch.empty(0, device=device),
                "values": torch.empty(0, device=device),
                "rewards": torch.empty(0, device=device),
            }
        # Merge
        full_ids     = torch.cat([b["full_ids"] for b in bufs], dim=0)
        prompt_lens  = torch.cat([b["prompt_lens"] for b in bufs], dim=0)
        old_logprobs = torch.cat([b["old_logprobs"] for b in bufs], dim=0)
        kls          = torch.cat([b["kls"] for b in bufs], dim=0)
        values       = torch.cat([b["values"] for b in bufs], dim=0)
        rewards      = torch.cat([b["rewards"] for b in bufs], dim=0)
        return {
            "full_ids": full_ids,
            "prompt_lens": prompt_lens,
            "old_logprobs": old_logprobs,
            "kls": kls,
            "values": values,
            "rewards": rewards,
        }

    # ------------------------ PPO Update ------------------------

    def train(self, prompt_iterator: Iterable[List[str]], total_rollouts: Optional[int] = None) -> Dict[str, float]:
        """Collect rollouts and perform PPO epochs with mini-batches."""
        self.model.train()
        tr_total = total_rollouts if total_rollouts is not None else self.cfg.total_rollouts

        # Value warmup (uses fresh mini-rollouts; only value_head updated)
        self._pretrain_value(_wrap_iterator(prompt_iterator), steps=self.cfg.pretrain_value_steps)

        # Collect on-policy rollouts
        buf = self.collect_rollouts(_wrap_iterator(prompt_iterator), cap=tr_total)
        if buf["full_ids"].size(0) == 0:
            return {"steps": 0}

        full_ids     = buf["full_ids"]
        prompt_lens  = buf["prompt_lens"]
        old_logprobs = buf["old_logprobs"]
        kls_store    = buf["kls"]
        values       = buf["values"]
        rewards      = buf["rewards"]

        # -------- Reward normalization + returns --------
        # normalize reward per buffer (z-score) and clip to reduce variance
        r = rewards
        r = (r - r.mean()) / (r.std() + 1e-8)
        r = torch.clamp(r, -3.0, 3.0)
        returns = r - self.cfg.beta_kl * kls_store           # [N]

        # Advantages
        advantages = returns - values.detach()                # [N]
        if self.cfg.normalize_advantages and advantages.numel() >= 4:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.clamp(advantages, -self.cfg.advantage_clip, self.cfg.advantage_clip)

        N = full_ids.size(0)
        indices = list(range(N))
        accum = 0

        huber = torch.nn.SmoothL1Loss()  # Huber for critic

        for _ in range(self.cfg.ppo_epochs):
            random.shuffle(indices)
            for start in range(0, N, self.cfg.mini_batch_size):
                mb_idx = indices[start : start + self.cfg.mini_batch_size]
                mb = torch.tensor(mb_idx, device=self.policy_device)

                mb_ids   = full_ids.index_select(0, mb)
                mb_plens = prompt_lens.index_select(0, mb)
                mb_old   = old_logprobs.index_select(0, mb)
                mb_advs  = advantages.index_select(0, mb)
                mb_vals  = values.index_select(0, mb)
                mb_rets  = returns.index_select(0, mb)

                # ---- Forward for logprobs + entropy + current values (per-sample loop keeps memory low) ----
                new_lps_list, ent_list, cur_vals_list = [], [], []
                for i in range(mb_ids.size(0)):
                    pl = int(mb_plens[i].item())
                    x = mb_ids[i : i + 1].to(self.policy_device)
                    with torch.autocast(device_type="cuda", dtype=self._amp_dtype) if self._amp_enabled else _nullctx():
                        logits_all = self.model(x[:, :-1]).logits
                    out = self._slice_resp_from_logits(logits_all, x, pl)
                    if out is None:
                        new_lps_list.append(torch.tensor(0.0, device=self.policy_device))
                        ent_list.append(torch.tensor(0.0, device=self.policy_device))
                    else:
                        resp_logits, resp_tokens = out
                        logp = torch.log_softmax(resp_logits, dim=-1)
                        tok_lp = torch.gather(logp, -1, resp_tokens.unsqueeze(-1)).squeeze(-1)  # [1, L]
                        new_lps_list.append(tok_lp.mean(dim=-1).squeeze(0))

                        probs = torch.softmax(resp_logits, dim=-1)
                        ent = -(probs * logp).sum(dim=-1).mean(dim=-1)  # [1]
                        ent_list.append(ent.squeeze(0))

                    # Prompt value (no lm_head)
                    cur_vals_list.append(self._prompt_value(x, pl).squeeze(0))

                new_lps   = torch.stack(new_lps_list, dim=0)  # [B_m]
                entropy   = torch.stack(ent_list, dim=0).mean()
                cur_vals  = torch.stack(cur_vals_list, dim=0)  # [B_m]

                # --- Recompute CURRENT KL for this minibatch (vs frozen ref) ---
                kl_list = []
                for i in range(mb_ids.size(0)):
                    pl = int(mb_plens[i].item())
                    x  = mb_ids[i : i + 1]
                    kl_i = self._resp_token_kl_mean(x, pl)  # current policy vs ref
                    kl_list.append(kl_i.squeeze(0))
                kl_current = torch.stack(kl_list, dim=0).mean()

                # Ratios and clipped policy loss + KL penalty
                ratios = torch.exp(new_lps - mb_old)
                eps = self.cfg.clip_epsilon
                surr1 = ratios * mb_advs
                surr2 = torch.clamp(ratios, 1.0 - eps, 1.0 + eps) * mb_advs
                policy_loss = -(torch.min(surr1, surr2).mean())
                policy_loss = policy_loss + self.cfg.beta_kl * kl_current  # KL penalty

                # Value loss with Huber and optional clipping
                if self.cfg.value_clip_epsilon and self.cfg.value_clip_epsilon > 0:
                    clipped = mb_vals + torch.clamp(cur_vals - mb_vals,
                                                    -self.cfg.value_clip_epsilon,
                                                    self.cfg.value_clip_epsilon)
                    v_loss_1 = huber(cur_vals, mb_rets)
                    v_loss_2 = huber(clipped,  mb_rets)
                    value_loss = torch.max(v_loss_1, v_loss_2)
                else:
                    value_loss = huber(cur_vals, mb_rets)

                total_loss = policy_loss + self.cfg.value_coeff * value_loss - self.cfg.entropy_coeff * entropy

                # Backprop with optional grad accumulation
                (total_loss / self.cfg.gradient_accumulation_steps).backward()
                accum += 1
                step_now = (accum % self.cfg.gradient_accumulation_steps == 0)

                if step_now:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.value_head.parameters()),
                        self.cfg.max_grad_norm
                    )
                    self.policy_optim.step()
                    self.value_optim.step()
                    self.policy_optim.zero_grad(set_to_none=True)
                    self.value_optim.zero_grad(set_to_none=True)

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    grad_norm = torch.tensor(float("nan"))

                # Adaptive β driven by CURRENT KL
                if self.cfg.adaptive_kl:
                    avg_kl = float(kl_current.detach().cpu())
                    if avg_kl > self.cfg.kl_target * 1.5:
                        self.cfg.beta_kl = min(self.cfg.beta_kl * 1.5, 5e-1)
                    elif avg_kl < self.cfg.kl_target / 1.5:
                        self.cfg.beta_kl = max(self.cfg.beta_kl / 1.5, 1e-4)
                else:
                    avg_kl = float("nan")

                # Logging
                self.global_step += 1
                if self.global_step % self.cfg.logging_steps == 0:
                    self.training_history.append({
                        "step": self.global_step,
                        "policy_loss": float(policy_loss.detach().cpu()),
                        "value_loss": float(value_loss.detach().cpu()),
                        "total_loss": float(total_loss.detach().cpu()),
                        "kl_divergence": float(avg_kl if math.isfinite(avg_kl) else 0.0),
                        "entropy": float(entropy.detach().cpu()),
                        "grad_norm": float(grad_norm.detach().cpu()) if torch.isfinite(grad_norm) else 0.0,
                        "beta_kl": float(self.cfg.beta_kl),
                    })

            if self.cfg.clear_cache_between_epochs and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Aggregate metrics
        if self.training_history:
            keys = ["policy_loss", "value_loss", "total_loss", "kl_divergence", "entropy", "grad_norm"]
            sums = {k: 0.0 for k in keys}
            for m in self.training_history:
                for k in keys:
                    sums[k] += m[k]
            n = len(self.training_history)
            out = {k: (sums[k] / n) for k in keys}
            out["steps"] = n
            return out
        else:
            return {"steps": 0}

    # ------------------------ Value warmup ------------------------

    def _pretrain_value(self, iterator: Iterable[List[str]], steps: int):
        """Quick value warmup using short fresh rollouts (no policy update)."""
        if steps <= 0:
            return
        it = iter(iterator)
        huber = torch.nn.SmoothL1Loss()
        for _ in range(steps):
            try:
                prompts = next(it)
            except StopIteration:
                break
            buf = self._collect_one_batch(prompts)
            if buf["full_ids"].numel() == 0:
                continue

            # Reward normalization (same as main)
            r = buf["rewards"]
            r = (r - r.mean()) / (r.std() + 1e-8)
            r = torch.clamp(r, -3.0, 3.0)
            returns = r - self.cfg.beta_kl * buf["kls"]  # [B]

            # current values (grad enabled) — CLONE to avoid 'inference' tensors
            v_list = []
            for i in range(buf["full_ids"].size(0)):
                pl = int(buf["prompt_lens"][i].item())
                seq = buf["full_ids"][i : i + 1].clone()
                v_list.append(self._prompt_value(seq, pl).squeeze(0))
            values = torch.stack(v_list, dim=0)  # [B]

            v_loss = huber(values, returns)
            self.value_optim.zero_grad(set_to_none=True)
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), self.cfg.max_grad_norm)
            self.value_optim.step()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ============================= Helpers =============================

class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *args): return False


def prompt_loader_from_dataset(dataset: Any, batch_size: int) -> Iterable[List[str]]:
    """
    Yields lists of prompt strings of size `batch_size`.
    Supports:
      • dataset.prompts : List[str]
      • iterable of dicts with "prompt"
      • iterable of raw strings
    """
    batch: List[str] = []
    if hasattr(dataset, "prompts"):
        for p in dataset.prompts:
            if p is None:
                continue
            batch.append(p)
            if len(batch) == batch_size:
                yield batch
                batch = []
    else:
        for ex in dataset:
            prompt = ex.get("prompt") if isinstance(ex, dict) else (ex if isinstance(ex, str) else None)
            if prompt is None:
                continue
            batch.append(prompt)
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def _wrap_iterator(it: Iterable[List[str]]) -> Iterable[List[str]]:
    """Return the iterator as-is (callers should pass a fresh iterator when needed)."""
    return it
