#!/usr/bin/env -S uv run python
"""Check transformers continuous batching prefix sharing for identical prompts (as in GRPO), on CPU.

Uses a tiny randomly initialized Qwen3 (scheduling logic is model-agnostic) and records, per scheduled
batch, how many prompt tokens each request actually prefilled, plus the "Found a prefix match" debug
messages. Prefix blocks are only shared once an earlier request has completed them, so copies scheduled in the
same batch prefill twice.

Usage: python benchmarks/prefix_sharing_test.py [prompt_len]
"""

import logging
import sys

import torch
from transformers import GenerationConfig, Qwen3Config, Qwen3ForCausalLM
from transformers.generation import ContinuousBatchingConfig
from transformers.generation.continuous_batching import scheduler as sched_mod

PROMPT_LEN, NUM_COPIES = int(sys.argv[1]) if len(sys.argv) > 1 else 5114, 4

logging.basicConfig(stream=sys.stdout, format="%(message)s")
cb_logger = logging.getLogger("ContinuousBatchingLogger")
cb_logger.setLevel(logging.DEBUG)

# Record, for every scheduled batch, the number of tokens each request contributes
BATCHES = []
_orig_process = sched_mod.Scheduler._process_candidates


def _process_candidates(self, candidates, token_budget, cache_budget, removed):
    out = _orig_process(self, candidates, token_budget, cache_budget, removed)
    BATCHES.append([(s.state.request_id, s.query_length) for s in out[0]] if out[0] else [])
    return out


sched_mod.Scheduler._process_candidates = _process_candidates

config = Qwen3Config(
    vocab_size=1000,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=16384,
)
torch.manual_seed(0)
model = Qwen3ForCausalLM(config).eval()
model.set_attn_implementation("sdpa")
prompt = torch.randint(0, 1000, (PROMPT_LEN,)).tolist()
gen_config = GenerationConfig(max_new_tokens=4, do_sample=False, eos_token_id=-1, pad_token_id=0)
cb_config = ContinuousBatchingConfig(use_cuda_graph=False, num_blocks=256, auto_switch_to_flash=False)
model.generate_batch(
    [prompt] * NUM_COPIES, generation_config=gen_config, continuous_batching_config=cb_config
)

print(f"\nprompt_len={PROMPT_LEN}, copies={NUM_COPIES}")
for i, batch in enumerate(BATCHES):
    if batch and any(q > 1 for _, q in batch):  # batches with prefill work
        print(f"batch {i}: {batch}")
prefill_tokens = sum(q for batch in BATCHES for _, q in batch if q > 1)
print(f"total prefill query tokens: {prefill_tokens} (no sharing would be {PROMPT_LEN * NUM_COPIES})")
