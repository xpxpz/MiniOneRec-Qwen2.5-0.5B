"""Assert checkpoint round-trips, updates and legal constrained predictions."""
import json
import math
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from lite_model import load_sft_adapter
from minionerec_trainer import normalize_group_rewards
from rl import ranking_rewards

root, info = map(Path, sys.argv[1:])
adapter_path = root / "sft/final_checkpoint"
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
merged = AutoModelForCausalLM.from_pretrained(adapter_path / "merged", torch_dtype="auto").eval()
adapter = load_sft_adapter(root / "base", adapter_path, tokenizer, merged.dtype).eval()
batch = tokenizer("test <a_0><b_0><c_0>", return_tensors="pt")
with torch.no_grad():
    # A BF16 merge rounds the low-rank update into each base weight.
    tolerance = 2e-3 if merged.dtype == torch.bfloat16 else 5e-6
    torch.testing.assert_close(adapter(**batch).logits, merged(**batch).logits, atol=tolerance, rtol=tolerance)
weights = adapter.state_dict()
assert any(torch.count_nonzero(v) for k, v in weights.items() if "lora_B" in k)
delta = next(v for k, v in weights.items() if "trainable_tokens_delta" in k)
base = next(v for k, v in weights.items() if "token_adapter.base_layer.weight" in k)
assert not torch.equal(base[adapter.peft_config["default"].trainable_token_indices], delta)

rl_weights = load_file(root / "grpo/final_checkpoint/adapter_model.safetensors")
assert any(torch.count_nonzero(v) for k, v in rl_weights.items() if "lora_B" in k)

catalog = {line.split("\t")[0] for line in info.read_text().splitlines()}
rows = json.loads((root / "predictions.json").read_text())
assert len(rows) == 19
for row in rows:
    assert len(row["predict"]) == 10
    assert all(candidate.strip() in catalog for candidate in row["predict"])

metrics = (root / "metrics.txt").read_text()
for field in ("HR", "NDCG"):
    actual = [float(token) for token in next(
        line for line in metrics.splitlines() if line.startswith(field)
    ).split("[")[1].split("]")[0].split()]
    for cutoff, value in zip((1, 3, 5, 10), actual, strict=True):
        total = 0.0
        for row in rows:
            target = row["output"]
            if isinstance(target, list):
                target = target[0]
            for rank, candidate in enumerate(row["predict"][:cutoff], start=1):
                if candidate.strip() == target.strip():
                    total += 1 if field == "HR" else 1 / math.log2(rank + 1)
                    break
        assert math.isclose(value, total / len(rows), abs_tol=1e-7)
print("SFT/SID/GRPO updates, checkpoint reload, catalog legality and ranking metrics verified.")

ranking, rule = ranking_rewards(4)
targets = ["A"] * 8
predictions = ["B", "A", "C", "D", "B", "C", "D", "E"]
binary = rule([], predictions, targets)
rank_scores = ranking([], predictions, targets)
assert binary == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
weights = [1 / math.log2(i + 2) for i in range(4)]
expected = [0.0 if i == 1 else -weight / sum(weights) for i, weight in enumerate(weights)]
torch.testing.assert_close(torch.tensor(rank_scores), torch.tensor(expected + [0.0] * 4))
reward = torch.tensor(binary) + torch.tensor(rank_scores)
advantages, _ = normalize_group_rewards(reward, 4)
assert advantages[1] > advantages[3] > advantages[2] > advantages[0]
torch.testing.assert_close(advantages[4:], torch.zeros(4))
assert math.isclose(advantages[:4].mean().item(), 0, abs_tol=1e-7)
print("Official rank reward, missed group behavior and normalized advantages verified.")
