#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

py="$PWD/.venv/bin/python"
[[ -x "$py" ]] || { echo "Run uv sync --locked first" >&2; exit 1; }
category=Industrial_and_Scientific
catalog="$PWD/data/processed/$category"
converted="$PWD/data/converted"
out="$PWD/outputs"
train="$converted/train/${category}_5_2016-10-2018-11.csv"
valid="$converted/valid/${category}_5_2016-10-2018-11.csv"
test="$converted/test/${category}_5_2016-10-2018-11.csv"
info="$converted/info/${category}_5_2016-10-2018-11.txt"
index="$catalog/$category.index.json"
meta="$catalog/$category.item.json"
emb="$catalog/$category.emb-qwen-td.npy"
sft="$out/sft/final_checkpoint/merged"
grpo="$out/grpo/final_checkpoint/merged"
export TZ=UTC WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
common=(--category "$category" --train_file "$train" --eval_file "$valid" --sid_index_path "$index" --item_meta_path "$meta")
mkdir -p "$out/logs"
stage="${1:-}"
case "$stage" in
    prepare)
        cmd=("$py" upstream/data/amazon18_data_process.py --dataset "$category" --st_year 1996 --st_month 10 --ed_year 2018 --ed_month 11 --output_path "$PWD/data/processed" --reviews_file "$PWD/data/raw/${category}_5.json.gz" --metadata_file "$PWD/data/raw/meta_${category}.json.gz");;
    embed)
        cmd=("$py" upstream/rq/text2emb/amazon_text2emb.py --dataset "$category" --root "$catalog" --plm_checkpoint "${EMBED_MODEL:-Qwen/Qwen3-Embedding-4B}" --batch_size "${EMBED_BATCH:-8}");;
    quantize)
        cmd=("$py" upstream/rq/rqvae.py --data_path "$emb" --ckpt_dir "$out/sid" --epochs 10000 --batch_size 20480 --lr 0.001);;
    index)
        cmd=("$py" upstream/rq/generate_indices.py --checkpoint "${2:?Specify best_collision_model.pth}" --output "$index");;
    convert)
        "$py" scripts/check_sid.py "$catalog" "$category"
        cmd=("$py" upstream/convert_dataset.py --dataset_name "$category" --data_dir "$catalog" --output_dir "$converted");;
    sft)
        cmd=("$py" upstream/sft.py "${common[@]}" --base_model "${QWEN_BASE:-Qwen/Qwen2.5-0.5B}" --batch_size 1024 --micro_batch_size "${SFT_MICRO_BATCH:-4}" --output_dir "$out/sft");;
    grpo)
        cmd=("$py" upstream/rl.py "${common[@]}" --model_path "$sft" --output_dir "$out/grpo" --info_file "$info" --train_batch_size 16 --eval_batch_size 16 --gradient_accumulation_steps 64 --num_train_epochs 2 --eval_step 0.0999 --beam_search True --num_generations 16 --reward_type ranking --test_during_training False --beta 0.001 --learning_rate 0.00001 --sync_ref_model True);;
    evaluate)
        model="${2:-$grpo}"
        cmd=("$py" upstream/evaluate.py --base_model "$model" --category "$category" --info_file "$info" --test_data_path "$test" --result_json_data "$out/test_predictions.json" --batch_size "${EVAL_BATCH:-1}")
        ;;
    *) echo "Usage: $0 {prepare|embed|quantize|index|convert|sft|grpo|evaluate}" >&2; exit 2;;
esac
case "$stage" in
    embed|quantize|index|sft|grpo|evaluate)
        [[ "${FORMAL_GPU_AUTHORIZED:-}" == 1 ]] || { echo "Formal GPU authorization required" >&2; exit 1; };;
esac
log="$(mktemp "$out/logs/$stage.XXXXXX")"
"$py" scripts/measure.py "$stage" "$out/logs" -- "${cmd[@]}" 2>&1 | tee "$log"
if [[ "$stage" == evaluate ]]; then
    "$py" upstream/calc.py --path "$out/test_predictions.json" --item_path "$info" | tee "$out/test_metrics.txt"
    "$py" scripts/audit_evaluation.py "$test" "$info" "$out/test_predictions.json" \
        "$out/test_metrics.txt" | tee "$out/test_audit.json"
fi
