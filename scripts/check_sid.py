"""Check that embedding row IDs, catalog keys, and SID assignments agree."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def check(directory, category):
    items = json.loads((directory / f"{category}.item.json").read_text())
    index = json.loads((directory / f"{category}.index.json").read_text())
    embeddings = np.load(directory / f"{category}.emb-qwen-td.npy", mmap_mode="r")
    if embeddings.ndim != 2 or embeddings.shape[0] == 0 or not np.isfinite(embeddings).all():
        raise ValueError(f"Invalid embedding matrix: {embeddings.shape}")
    expected = {str(i) for i in range(embeddings.shape[0])}
    if items.keys() != expected or index.keys() != expected:
        raise ValueError("Catalog, embedding rows and SID keys differ")
    for key, sid in index.items():
        if len(sid) != 3 or any(
            re.fullmatch(rf"<{prefix}_[0-9]+>", token) is None
            for prefix, token in zip("abc", sid)
        ):
            raise ValueError(f"Malformed SID for item {key}: {sid}")
    counts = Counter(tuple(sid) for sid in index.values())
    print(json.dumps({
        "embedding_shape": embeddings.shape,
        "items": len(items),
        "distinct_sids": len(counts),
        "collision_rate": 1 - len(counts) / len(items),
        "largest_sid_group": max(counts.values()),
    }))


if __name__ == "__main__":
    check(Path(sys.argv[1]), sys.argv[2])
