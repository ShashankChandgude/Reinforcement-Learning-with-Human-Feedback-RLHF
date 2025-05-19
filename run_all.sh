#!/usr/bin/env bash
set -e

echo "1. Supervised fine-tuning"
python main.py --phase sft    --config configs/sft.yaml

echo "2. Reward-model training"
python main.py --phase reward --config configs/reward.yaml

echo "3. PPO training"
python main.py --phase ppo    --config configs/ppo.yaml

echo "4. In-distribution evaluation"
python eval_runner.py --phase evaluate --config configs/evaluate_indist.yaml

echo "5. OOD evaluation"
python eval_runner.py --phase evaluate --config configs/evaluate_ood.yaml

echo "All done! Reports in evaluation/*.json"
