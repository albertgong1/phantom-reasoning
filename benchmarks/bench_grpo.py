"""Run a GRPO training script unchanged, marking each generation call with an NVTX range for Nsight Systems.

The CUDA profiler capture range covers training steps PROFILE_START..PROFILE_END (inclusive), so nsys
should be launched with `--capture-range=cudaProfilerApi`. NVTX ranges are host-side markers (no
synchronization), and cudaProfilerStart/Stop are no-ops when no profiler is attached, so the wrapper is also
safe to run without nsys.

This wraps `GRPOTrainer._generate_single_turn` at runtime; it does not modify the training code.

Usage (see run_benchmarks.sh):
    GRPO_PY=src/phantom_reasoner/grpo.py nsys profile --capture-range=cudaProfilerApi ... \
        accelerate launch ... benchmarks/bench_grpo.py --config <recipe.yaml>
"""

import faulthandler
import os
import runpy
import signal
import time

import torch
from trl.trainer.grpo_trainer import GRPOTrainer

GRPO_PY = os.environ.get("GRPO_PY", os.path.join(os.getcwd(), "src/phantom_reasoner/grpo.py"))
PROFILE_START = int(os.environ.get("PROFILE_START", 2))
PROFILE_END = int(os.environ.get("PROFILE_END", 4))
STATE = {"profiling": False}

# torch.compile specializes on each decoder layer's integer `layer_idx`, so compiling every layer with
# attention included needs a recompile limit of about layers x batch sizes (the default 8 is far too low)
if "DYNAMO_RECOMPILE_LIMIT" in os.environ:
    torch._dynamo.config.recompile_limit = int(os.environ["DYNAMO_RECOMPILE_LIMIT"])

# Alternatively, keep FlashAttention out of torch.compile: it is already a single fused kernel, and the
# paged cache indexes a Python list by `layer_idx` inside it, which forces one compiled copy per layer.
# Models look the function up in ALL_ATTENTION_FUNCTIONS on every forward, so that entry is replaced.
if os.environ.get("DYNAMO_DISABLE_ATTN") == "1":
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    for _impl in ("flash_attention_2", "flash_attention_3", "flash_attention_4"):
        ALL_ATTENTION_FUNCTIONS._global_mapping[_impl] = torch._dynamo.disable(
            ALL_ATTENTION_FUNCTIONS._global_mapping[_impl]
        )

# Continuous batching diagnostics (transformers >= 5.x only):
#   CB_PERSISTENT=1 keeps the manager (paged cache, CUDA graphs, warmup) alive across generate_batch calls
#   CB_TRACE=1 logs, per generation call, the prompt lengths and the prefill query tokens actually scheduled
CB_STATS = {"batches": 0, "prefill_tokens": 0, "prompt_lens": []}
if os.environ.get("CB_PERSISTENT") == "1" or os.environ.get("CB_TRACE") == "1":
    from transformers.generation.continuous_batching import continuous_api
    from transformers.generation.continuous_batching import scheduler as cb_scheduler

    _orig_generate_batch = continuous_api.ContinuousMixin.generate_batch

    def _generate_batch(self, inputs, *args, **kwargs):
        if os.environ.get("CB_PERSISTENT") == "1":
            kwargs["persistent_manager"] = True
            # Like TRL's vLLM path (reset_prefix_cache after each weight sync), drop prefix-cached KV blocks:
            # they were computed with the previous step's weights. CB_KEEP_PREFIX_CACHE=1 keeps them (stale).
            manager = getattr(self, "_cached_continuous_batching_manager", None)
            if manager is not None and manager.batch_processor is not None:
                if os.environ.get("CB_KEEP_PREFIX_CACHE") != "1":
                    manager.batch_processor.cache.reset()
        CB_STATS["prompt_lens"] = [len(x) for x in inputs]
        CB_STATS["unique_prompt_tokens"] = sum(len(x) for x in set(map(tuple, inputs)))
        return _orig_generate_batch(self, inputs, *args, **kwargs)

    continuous_api.ContinuousMixin.generate_batch = _generate_batch

    _orig_process = cb_scheduler.Scheduler._process_candidates

    def _process_candidates(self, *args, **kwargs):
        out = _orig_process(self, *args, **kwargs)
        if out[0]:
            CB_STATS["batches"] += 1
            CB_STATS["prefill_tokens"] += sum(s.query_length for s in out[0] if s.query_length > 1)
        return out

    cb_scheduler.Scheduler._process_candidates = _process_candidates

# `kill -USR1 <pid>` dumps every Python thread's stack to stderr (for diagnosing hangs)
faulthandler.register(signal.SIGUSR1, all_threads=True)

_orig_generate = GRPOTrainer._generate_single_turn


def _generate_single_turn(self, *args, **kwargs):
    step = self.state.global_step
    # Generation is the first thing in each step, so starting/stopping here brackets whole steps
    if step == PROFILE_START and not STATE["profiling"]:
        torch.cuda.profiler.start()
        STATE["profiling"] = True
    elif step == PROFILE_END + 1 and STATE["profiling"]:
        torch.cuda.profiler.stop()
        STATE["profiling"] = False
    torch.cuda.nvtx.range_push(f"gen step {step}")
    CB_STATS.update(batches=0, prefill_tokens=0)
    start = time.perf_counter()
    try:
        return _orig_generate(self, *args, **kwargs)
    finally:
        torch.cuda.nvtx.range_pop()
        # Generation returns token ids to the CPU, so the GPU work is done by the time it returns
        msg = f"[bench] step {step} gen_s={time.perf_counter() - start:.2f}"
        if CB_STATS["prompt_lens"]:
            lens = CB_STATS["prompt_lens"]
            msg += (
                f" prompts={len(lens)} unique_prompt_tokens={CB_STATS['unique_prompt_tokens']}"
                f" all_prompt_tokens={sum(lens)}"
                f" prefill_tokens={CB_STATS['prefill_tokens']} batches={CB_STATS['batches']}"
            )
        print(msg, flush=True)


GRPOTrainer._generate_single_turn = _generate_single_turn

try:
    runpy.run_path(GRPO_PY, run_name="__main__")
finally:
    if STATE["profiling"]:
        torch.cuda.profiler.stop()
