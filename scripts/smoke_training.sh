#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
py="${PYTHON:-.venv/bin/python}"
work="$(mktemp -d)"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false USE_TF=0
PYTHON="$py" scripts/smoke_sid.sh "$work"
"$py" scripts/make_smoke_model.py "$work/base"
cat=Industrial_and_Scientific
name="${cat}_5_2016-10-2018-11"
data="$work/converted"
mkdir "$work/encoding"
cp "$work/$cat/$cat.item.json" "$work/encoding/"
"$py" upstream/rq/text2emb/amazon_text2emb.py --dataset "$cat" --root "$work/encoding" --plm_checkpoint "$work/base" --batch_size 4 --max_sent_len 64
"$py" - "$work/encoding/$cat.emb-qwen-td.npy" <<'PYCODE'
import sys
import numpy as np
emb = np.load(sys.argv[1])
assert emb.shape == (32, 24) and np.isfinite(emb).all()
PYCODE
info="$data/info/$name.txt"
sft="$work/sft/final_checkpoint"
grpo="$work/grpo/final_checkpoint"
common=(--category "$cat" --train_file "$data/train/$name.csv" --eval_file "$data/valid/$name.csv" --sid_index_path "$work/$cat/$cat.index.json" --item_meta_path "$work/$cat/$cat.item.json" --lora_rank 2 --sample 8)
"$py" upstream/sft.py "${common[@]}" --base_model "$work/base" --output_dir "$work/sft" --micro_batch_size 2 --batch_size 2 --max_steps 2
"$py" upstream/rl.py "${common[@]}" --model_path "$sft/merged" --output_dir "$work/grpo" --info_file "$info" --train_batch_size 16 --eval_batch_size 16 --num_generations 16 --max_steps 8 --beam_search True --reward_type ranking --test_during_training False --sync_ref_model True --ref_sync_steps 2 --beta 0.001
"$py" upstream/evaluate.py --base_model "$grpo/merged" --category "$cat" --info_file "$info" --test_data_path "$data/test/$name.csv" --result_json_data "$work/predictions.json" --num_beams 10 --max_new_tokens 8
"$py" upstream/calc.py --path "$work/predictions.json" --item_path "$info" | tee "$work/metrics.txt"
PYTHONPATH="upstream${PYTHONPATH:+:$PYTHONPATH}" "$py" scripts/check_smoke_training.py "$work" "$info"
