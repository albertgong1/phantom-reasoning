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

import torch
from trl.trainer.grpo_trainer import GRPOTrainer

GRPO_PY = os.environ.get("GRPO_PY", os.path.join(os.getcwd(), "src/phantom_reasoner/grpo.py"))
PROFILE_START = int(os.environ.get("PROFILE_START", 2))
PROFILE_END = int(os.environ.get("PROFILE_END", 4))
STATE = {"profiling": False}

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
    try:
        return _orig_generate(self, *args, **kwargs)
    finally:
        torch.cuda.nvtx.range_pop()


GRPOTrainer._generate_single_turn = _generate_single_turn

try:
    runpy.run_path(GRPO_PY, run_name="__main__")
finally:
    if STATE["profiling"]:
        torch.cuda.profiler.stop()
