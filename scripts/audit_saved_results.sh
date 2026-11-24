#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python="${PYTHON:-python3}"
base=outputs/results
for model in sft grpo; do
  "$python" scripts/audit_evaluation.py "$base/audit_inputs/test.csv" "$base/audit_inputs/catalog.txt" "$base/${model}_test_predictions.json" "$base/${model}_test_metrics.txt" > /dev/null
  printf '%s: audit passed\n' "$model"
done
