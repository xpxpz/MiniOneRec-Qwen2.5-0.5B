"""Create a tiny CPU-only Qwen2 checkpoint with an offline test tokenizer."""
import sys

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

torch.manual_seed(42)
destination = sys.argv[1]
tokens = ["<unk>", "<eos>"] + sorted(pre_tokenizers.ByteLevel.alphabet())
raw = Tokenizer(models.BPE({token: idx for idx, token in enumerate(tokens)}, [], unk_token="<unk>"))
raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
raw.decoder = decoders.ByteLevel()
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>"
)
tokenizer.add_tokens(["###", " Response", ":\n"])
assert len(tokenizer("### Response:\n").input_ids) == 3
tokenizer.save_pretrained(destination)
Qwen2ForCausalLM(Qwen2Config(
    vocab_size=len(tokenizer), hidden_size=24, intermediate_size=32,
    num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
    max_position_embeddings=2048, eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)).save_pretrained(destination)
