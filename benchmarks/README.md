# Generation engine benchmarks

Compares GRPO generation with vLLM (colocate) against transformers continuous batching (`generate_batch`) and
padded `generate()`, on the single-GPU Qwen3-0.6B PhantomWiki smoke config.

| File                     | Purpose                                                                                         |
| ------------------------ | ----------------------------------------------------------------------------------------------- |
| `run_benchmarks.sh`      | Runs each variant (optionally under Nsight Systems) and exports traces to SQLite                |
| `bench_grpo.py`          | Runs `grpo.py` unchanged, wrapping generation in NVTX ranges and starting/stopping the profiler |
| `analyze_nsys.py`        | Splits GPU kernel time into generation vs training and attention vs other kernels               |
| `prefix_sharing_test.py` | CPU check of how much prompt prefill continuous batching shares across identical prompts        |

## Environments

vLLM and the latest transformers can't share an env (vLLM pins torch and caps transformers), so there are two:

- `phantom-reasoning` (vLLM variant): `main`'s `pyproject.toml` (vLLM 0.13, transformers 4.57, pinned TRL).
- `phantom-reasoning-tf` (all other variants): the `transformers-cb` branch's `pyproject.toml` (transformers
  5.20.2, TRL 1.16.0, torch 2.9.0, flash-attn 2.8.3, no vLLM).

The configs are `recipes/Qwen/Qwen3-0.6B/grpo/config_pw_1gpu_smoke{,_cb,_generate}.yaml`.

## Usage

```bash
benchmarks/run_benchmarks.sh vllm cb cb_compile cb_nograph generate
python benchmarks/analyze_nsys.py runs/benchmarks/{vllm,cb,cb_compile,cb_nograph,generate}.sqlite

# 32 completions per step; nsys with CUDA-graph node tracing hangs on continuous batching at this size
benchmarks/run_benchmarks.sh vllm_n32
benchmarks/run_benchmarks.sh --no-nsys cb_n32
```

Outputs (`<variant>.log`, `.nsys-rep`, `.sqlite`) go to `runs/benchmarks/`. `step_time` per step is in each log.

## Results (one 24GB GPU)

4 completions per step (step 4, under nsys):

| Variant                      | Generation wall | Throughput | `step_time` | Peak memory |
| ---------------------------- | --------------- | ---------- | ----------- | ----------- |
| vLLM colocate                | 8.7 s           | ~470 tok/s | 12.7 s      | 22.2 GB     |
| CB + CUDA graphs + compile 1 | 13.8 s          | ~300 tok/s | 16.9 s      |             |
| CB + CUDA graphs (default)   | 16.4 s          | ~215 tok/s | 20.0 s      | 13.7 GB     |
| CB, no CUDA graphs           | 69.9 s          | ~60 tok/s  | 72.9 s      |             |
| Padded `generate()`          | 76.8 s          | ~53 tok/s  | 79.9 s      |             |

32 completions per step:

| Variant                    | Generation wall       | `step_time` | 10 steps | Notes                            |
| -------------------------- | --------------------- | ----------- | -------- | -------------------------------- |
| vLLM colocate              | 18.3 s (~1,700 tok/s) | ~55 s       | 583 s    | nsys; `expandable_segments:True` |
| CB + CUDA graphs (default) | ~40 s (estimated)     | ~77 s       | 763 s    | no profiler                      |
| Padded `generate()`        |                       |             |          | OOM at step 0                    |

Caveats: vLLM ran with transformers 4.57 / the pinned TRL; `cb_compile` here hit dynamo's recompile limit (8), so
only ~8 of 28 layers compiled. `run_benchmarks.sh` now excludes FlashAttention from compilation for `cb_compile`
(`DYNAMO_DISABLE_ATTN=1` in `bench_grpo.py`), which removes the per-layer recompiles. Continuous batching's
generation time at 32 completions is `step_time` minus vLLM's ~37 s of training.
