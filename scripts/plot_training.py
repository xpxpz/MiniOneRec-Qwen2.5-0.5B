"""Render training curves from a snapshot of the formal stage logs."""
import argparse
import ast
import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def latest(logs, stage):
    return max(logs.glob(f"{stage}.*"), default=None, key=lambda f: f.stat().st_mtime_ns)


def read_log(path):
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", path.read_text(errors="replace"))


def rq_rows(log):
    rows = {}
    for epoch, loss, recon in re.findall(
        r"epoch (\d+) training.*?train loss:\s*([\d.eE+-]+).*?reconstruction loss:\s*([\d.eE+-]+)", log
    ):
        rows[int(epoch) + 1] = {"train_loss": float(loss), "reconstruction_loss": float(recon)}
    for epoch, collision in re.findall(r"epoch (\d+) evaluating.*?collision_rate:\s*([\d.eE+-]+)", log):
        rows.setdefault(int(epoch) + 1, {})["collision_rate"] = float(collision)
    return [{"epoch": epoch, **row} for epoch, row in sorted(rows.items())]


def trainer_rows(log):
    for line in log.splitlines():
        if not line.startswith("{'") or not line.endswith("}"):
            continue
        try:
            row = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(row.get("epoch"), (int, float)):
            continue
        yield {k: v for k, v in row.items() if type(v) in (int, float) and math.isfinite(v)}


def save_curves(rows, panels, destination, title):
    def present(key):
        return any(key in row for row in rows)

    panels = [panel for panel in panels if any(present(k) for k, _ in panel[1])]
    if not panels:
        return
    fig, axes = plt.subplots(len(panels), figsize=(9.5, 2.85 * len(panels)),
                             sharex=True, squeeze=False, layout="constrained")
    for axis, (label, series) in zip(axes[:, 0], panels):
        for key, display in series:
            points = [(row["epoch"], row[key]) for row in rows if key in row]
            if points:
                x, y = zip(*points)
                axis.plot(x, y, label=display, linewidth=1.45,
                          marker="o" if len(points) < 30 else None, markersize=3)
        if title == "RQ-VAE" and label == "Loss":
            axis.set_yscale("log")
            label = "Loss (log scale)"
        axis.set_ylabel(label)
        axis.grid(alpha=0.22)
        axis.legend(loc="best", frameon=False)
    axes[-1, 0].set_xlabel("Epoch")
    fig.suptitle(f"MiniOneRec-Lite | {title}", fontweight="bold")
    fig.savefig(destination, dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stages = [
        ("quantize", "RQ-VAE", rq_rows, [
            ("Loss", [("train_loss", "Training"), ("reconstruction_loss", "Reconstruction")]),
            ("Collision rate", [("collision_rate", "Validation")]),
        ]),
        ("sft", "LoRA SFT", trainer_rows, [
            ("Loss", [("loss", "Training"), ("eval_loss", "Validation")]),
            ("Gradient norm", [("grad_norm", "Training")]),
            ("Learning rate", [("learning_rate", "Training")]),
        ]),
        ("grpo", "LoRA GRPO", trainer_rows, [
            ("Loss", [("loss", "Training"), ("eval_loss", "Validation")]),
            ("Reward", [("reward", "Training"), ("eval_reward", "Validation")]),
            ("KL", [("kl", "Training"), ("eval_kl", "Validation")]),
            ("Beam diversity", [("categorical_diversity", "Training"),
                                ("eval_categorical_diversity", "Validation")]),
        ]),
    ]
    for stage, title, parse, panels in stages:
        log = latest(args.logs, stage)
        if log is None:
            continue
        rows = list(parse(read_log(log)))
        if not rows:
            continue
        fields = ["epoch", *sorted(set().union(*(row.keys() for row in rows)) - {"epoch"})]
        with (args.output / f"{stage}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        save_curves(rows, panels, args.output / f"{stage}.png", title)
        print(f"{log.name}: {len(rows)} rows, last epoch {rows[-1]['epoch']}")


if __name__ == "__main__":
    main()
