"""Audit prediction coverage and independently reproduce the official SID metrics."""
import argparse
import ast
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re


def normalize(sid):
    return sid.strip(' "\r\n\t')


def audit(test_path, catalog_path, predictions_path, metrics_path, expected_rows, beams):
    with test_path.open(newline="") as stream:
        targets = [normalize(row["item_sid"]) for row in csv.DictReader(stream)]
    catalog = Counter(line.split("\t", 1)[0].strip()
                      for line in catalog_path.read_text().splitlines())
    predictions = json.loads(predictions_path.read_text())
    if len(targets) != expected_rows or len(predictions) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(targets)} test rows and "
                         f"{len(predictions)} prediction rows")
    cutoffs = [k for k in (1, 3, 5, 10, 20, 50) if k <= beams]
    hits = dict.fromkeys(cutoffs, 0)
    discounted = dict.fromkeys(cutoffs, 0.0)
    illegal = duplicate_beams = 0
    for i, (target, row) in enumerate(zip(targets, predictions, strict=True)):
        output = row["output"]
        candidates = [normalize(sid) for sid in row["predict"]]
        if normalize(output[0] if isinstance(output, list) else output) != target or target not in catalog:
            raise ValueError(f"Test target mismatch or missing catalog SID at row {i + 1}")
        if len(candidates) != beams:
            raise ValueError(f"Expected {beams} beams at row {i + 1}, got {len(candidates)}")
        illegal += sum(candidate not in catalog for candidate in candidates)
        duplicate_beams += len(candidates) - len(set(candidates))
        if target in candidates:
            rank = candidates.index(target) + 1
            for cutoff in cutoffs:
                if rank <= cutoff:
                    hits[cutoff] += 1
                    discounted[cutoff] += 1 / math.log2(rank + 1)
    raw_metrics = metrics_path.read_text()
    official_cutoffs = re.search(r"^\[([\d, ]+)\]$", raw_metrics, re.MULTILINE)
    if official_cutoffs is None or ast.literal_eval(official_cutoffs.group(0)) != cutoffs:
        raise ValueError("Official metric cutoffs do not match the requested beams")
    result = {
        "test_rows": len(targets), "beams_per_row": beams, "illegal_beams": illegal,
        "duplicate_beams": duplicate_beams, "catalog_items": sum(catalog.values()),
        "unique_catalog_sids": len(catalog),
    }
    for metric, totals in (("HR", hits), ("NDCG", discounted)):
        expected = [totals[k] / len(targets) for k in cutoffs]
        match = re.search(rf"^{metric}[:\t ]+\[([^\]]+)\]", raw_metrics, re.MULTILINE)
        if match is None:
            raise ValueError(f"Missing official {metric}")
        values = [float(raw) for raw in match.group(1).split()]
        if len(values) != len(expected) or any(
            not math.isclose(x, y, rel_tol=0, abs_tol=1e-7) for x, y in zip(values, expected)
        ):
            raise ValueError(f"Official {metric} does not match independently recomputed values")
        result[metric] = dict(zip(cutoffs, expected, strict=True))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("test", "catalog", "predictions", "metrics"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--rows", type=int, default=4533)
    parser.add_argument("--beams", type=int, default=50)
    args = parser.parse_args()
    result = audit(args.test, args.catalog, args.predictions, args.metrics, args.rows, args.beams)
    print(json.dumps(result, indent=2))
    if result["illegal_beams"]:
        raise SystemExit("Constrained decoding produced SIDs outside the catalog")
