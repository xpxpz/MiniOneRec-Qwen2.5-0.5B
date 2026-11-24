#!/usr/bin/env bash
set -euo pipefail
[[ "${FORMAL_GPU_AUTHORIZED:-}" == 1 ]] || { echo "Formal GPU authorization required" >&2; exit 1; }
cd "$(dirname "$0")/.."
export MINIONEREC_ATTENTION=eager CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 WANDB_MODE=disabled
base="$PWD/outputs/sft/final_checkpoint/merged"
model="$base"
category=Industrial_and_Scientific
data="$PWD/data/converted"
catalog="$PWD/data/processed/$category"
name="${category}_5_2016-10-2018-11"
mkdir -p outputs/logs
.venv/bin/python scripts/measure.py grpo_restart outputs/logs -- .venv/bin/python upstream/rl.py --category "$category" --train_file "$data/train/$name.csv" --eval_file "$data/valid/$name.csv" --sid_index_path "$catalog/$category.index.json" --item_meta_path "$catalog/$category.item.json" --info_file "$data/info/$name.txt" --model_path "$model" --output_dir "$PWD/outputs/grpo-restart" --train_batch_size 16 --eval_batch_size 16 --gradient_accumulation_steps 64 --num_train_epochs 2 --eval_step 0.0999 --beam_search True --num_generations 16 --reward_type ranking --test_during_training False --beta 0.005 --learning_rate 0.000005 --sync_ref_model True
FORMAL_GPU_AUTHORIZED=1 scripts/pipeline.sh evaluate "$PWD/outputs/grpo-restart/final_checkpoint/merged"
