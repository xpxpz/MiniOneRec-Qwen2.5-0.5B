"""PEFT setup shared by the Lite training stages."""

import math

from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, TrainerCallback


def with_lora(model, rank, token_ids=None):
    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=rank,
        lora_alpha=2 * rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        trainable_token_indices=token_ids,
    )
    return get_peft_model(model, config)


def export_merged(adapter, tokenizer, destination):
    merged = adapter.merge_and_unload()
    merged.config.use_cache = True
    merged.generation_config.use_cache = True
    merged.save_pretrained(destination)
    tokenizer.save_pretrained(destination)


def load_sft_adapter(base_path, adapter_path, tokenizer, dtype):
    base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=dtype)
    base.resize_token_embeddings(len(tokenizer))
    return PeftModel.from_pretrained(base, adapter_path)

class FiniteTrainingMetrics(TrainerCallback):
    """Stop before saving a checkpoint from a non-finite training run."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        for key in ("loss", "grad_norm", "eval_loss", "reward", "kl"):
            value = (logs or {}).get(key)
            if value is not None and not math.isfinite(value):
                raise FloatingPointError(f"{key}={value} at optimizer step {state.global_step}")
