#!/usr/bin/env -S uv run python
"""Measure how much prompt prefill transformers continuous batching shares across identical prompts on GPU.

With CUDA graphs and FlashAttention, use_async_batching defaults to on. Async batching schedules batch
N+1 before batch N's completed blocks are registered for prefix sharing, so GRPO's copies of a prompt
never share. A smaller max_batch_tokens spreads the copies over more batches, so later copies reuse
earlier ones' blocks. Uses a tiny randomly initialized Qwen3 (scheduling is model-agnostic).

Usage: python benchmarks/prefix_sharing_gpu_test.py [prompt_len] [copies]
"""

import sys

import torch
from transformers import GenerationConfig, Qwen3Config, Qwen3ForCausalLM
from transformers.generation import ContinuousBatchingConfig
from transformers.generation.continuous_batching import scheduler as sched_mod

PROMPT_LEN = int(sys.argv[1]) if len(sys.argv) > 1 else 5132
COPIES = int(sys.argv[2]) if len(sys.argv) > 2 else 4

PREFILL_TOKENS = []
_orig_process = sched_mod.Scheduler._process_candidates


def _process_candidates(self, *args, **kwargs):
    out = _orig_process(self, *args, **kwargs)
    if out[0]:
        PREFILL_TOKENS.append(sum(s.query_length for s in out[0] if s.query_length > 1))
    return out


sched_mod.Scheduler._process_candidates = _process_candidates

config = Qwen3Config(
    vocab_size=1000,
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    max_position_embeddings=16384,
)
torch.manual_seed(0)
model = Qwen3ForCausalLM(config).to("cuda", torch.bfloat16).eval()
model.set_attn_implementation("flash_attention_2")
prompt = torch.randint(0, 1000, (PROMPT_LEN,)).tolist()
gen_config = GenerationConfig(max_new_tokens=8, do_sample=False, eos_token_id=-1, pad_token_id=0)
ideal = PROMPT_LEN + (COPIES - 1) * (PROMPT_LEN % 256)
print(
    f"prompt_len={PROMPT_LEN}, copies={COPIES}: "
    f"no sharing = {PROMPT_LEN * COPIES}, ideal ~ {ideal} prefill tokens"
)
for use_async in (None, False):
    for max_batch_tokens in (8192, 4096, 2048, 1024):
        PREFILL_TOKENS.clear()
        cb_config = ContinuousBatchingConfig(
            num_blocks=256, use_async_batching=use_async, max_batch_tokens=max_batch_tokens
        )
        model.generate_batch(
            [prompt] * COPIES, generation_config=gen_config, continuous_batching_config=cb_config
        )
        label = "default (on)" if use_async is None else "off"
        print(
            f"  async {label:12s} max_batch_tokens={max_batch_tokens:5d}: "
            f"{sum(PREFILL_TOKENS):6d} prefill tokens"
        )
