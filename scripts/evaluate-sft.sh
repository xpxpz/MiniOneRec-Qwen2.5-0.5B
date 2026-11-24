#!/usr/bin/env bash
set -euo pipefail
[[ "${FORMAL_GPU_AUTHORIZED:-}" == 1 ]] || { echo "Formal GPU authorization required" >&2; exit 1; }
cd "$(dirname "$0")/.."
export MINIONEREC_ATTENTION=eager CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled
category=Industrial_and_Scientific
info="$PWD/data/converted/info/${category}_5_2016-10-2018-11.txt"
test="$PWD/data/converted/test/${category}_5_2016-10-2018-11.csv"
.venv/bin/python scripts/measure.py evaluate_sft outputs/logs -- .venv/bin/python upstream/evaluate.py --base_model "$PWD/outputs/sft/final_checkpoint/merged" --category "$category" --info_file "$info" --test_data_path "$test" --result_json_data "$PWD/outputs/sft_test_predictions.json" --batch_size 1
.venv/bin/python upstream/calc.py --path outputs/sft_test_predictions.json --item_path "$info" > outputs/sft_test_metrics.txt
.venv/bin/python scripts/audit_evaluation.py "$test" "$info" outputs/sft_test_predictions.json outputs/sft_test_metrics.txt > outputs/sft_test_audit.json
