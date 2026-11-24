#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
py="${PYTHON:-.venv/bin/python}"
work="${1:-$(mktemp -d)}"
mkdir -p "$work"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "Smoke outputs: $work"
"$py" - "$work" <<'PY'
import gzip
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
with gzip.open(root / "reviews.json.gz", "wt") as reviews, gzip.open(root / "meta.json.gz", "wt") as meta:
    for item in range(32):
        print(json.dumps({"asin": f"P{item}", "title": f"Industrial part {item}", "description": ["Test part"]}), file=meta)
        for user in range(6):
            print(json.dumps({"asin": f"P{item}", "reviewerID": f"U{user}", "unixReviewTime": 1451606400 + item * 86400 + user, "overall": 5, "reviewText": "fixture"}), file=reviews)
PY
"$py" upstream/data/amazon18_data_process.py --dataset Industrial_and_Scientific --reviews_file "$work/reviews.json.gz" --metadata_file "$work/meta.json.gz" --output_path "$work"
catalog="$work/Industrial_and_Scientific"
emb="$catalog/Industrial_and_Scientific.emb-qwen-td.npy"
"$py" - "$emb" <<'PY'
import sys
import numpy as np
np.save(sys.argv[1], np.random.default_rng(42).normal(size=(32, 16)).astype("float32"))
PY
"$py" upstream/rq/rqvae.py --data_path "$emb" --ckpt_dir "$work/sid" --device cpu --epochs 2 --eval_step 1 --num_workers 0 --num_emb_list 4 4 4 --e_dim 4 --layers 8 --batch_size 32 --kmeans_iters 5
checkpoint="$(find "$work/sid" -name best_collision_model.pth)"
"$py" upstream/rq/generate_indices.py --checkpoint "$checkpoint" --device cpu --output "$catalog/Industrial_and_Scientific.index.json"
"$py" scripts/check_sid.py "$catalog" Industrial_and_Scientific
"$py" upstream/convert_dataset.py --data_dir "$catalog" --output_dir "$work/converted"
"$py" - "$work/converted" <<'PY'
import csv
import sys
from pathlib import Path
root = Path(sys.argv[1])
for split, count in [("train", 148), ("valid", 19), ("test", 19)]:
    with next((root / split).glob("*.csv")).open() as stream:
        assert len(list(csv.DictReader(stream))) == count, split
PY
