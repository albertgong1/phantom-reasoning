#!/usr/bin/env bash
# Benchmark GRPO generation engines on the single-GPU Qwen3-0.6B smoke config, optionally under Nsight Systems
# (training steps PROFILE_START..PROFILE_END, default 2..4), then export each trace to SQLite for analyze_nsys.py.
#
# Variants:
#   vllm       : vLLM colocate, in $VLLM_ENV (vLLM 0.13, transformers 4.57) with main's grpo.py
#   cb         : transformers continuous batching (CUDA graphs on, the default), in $TF_ENV (transformers 5.20)
#   cb_nograph : continuous batching with CUDA graphs off
#   cb_compile : continuous batching with default_compile_level=1 (torch.compile on the decode path), with
#                FlashAttention excluded from compilation so all layers share one compiled graph
#   cb_kv<N>   : continuous batching with kv_padding_interval_size=N tokens (default 16384), e.g. cb_kv1024
#   generate   : padded model.generate() with a dynamic KV cache
# Suffix _n32 runs 32 completions per step (4 prompts x 8 generations, via gradient accumulation; the training
# micro-batch stays at 4).
#
# nsys with --cuda-graph-trace=node hung / segfaulted on cb_n32 and cb_nograph_n32; use --no-nsys for those.
#
# Usage: benchmarks/run_benchmarks.sh [--no-nsys] <variant> [<variant> ...]
#   e.g. benchmarks/run_benchmarks.sh vllm cb cb_compile generate
#        benchmarks/run_benchmarks.sh --no-nsys cb_n32
# Env: VLLM_ENV, TF_ENV (conda env names), CONDA_ENVS (default ~/.conda/envs),
#      OUT_DIR (default runs/benchmarks), NSYS (default /usr/local/cuda/bin/nsys)

use_nsys=true
if [[ $1 == --no-nsys ]]; then
    use_nsys=false
    shift
fi
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 [--no-nsys] <vllm|cb|cb_nograph|cb_compile|generate>[_n32] ..."
    exit 1
fi

REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
VLLM_ENV=${VLLM_ENV:-phantom-reasoning}
TF_ENV=${TF_ENV:-phantom-reasoning-tf}
OUT_DIR=${OUT_DIR:-$REPO/runs/benchmarks}
NSYS=${NSYS:-/usr/local/cuda/bin/nsys}
R=recipes/Qwen/Qwen3-0.6B/grpo
CONDA_ENVS=${CONDA_ENVS:-$HOME/.conda/envs}

cd "$REPO" || exit 1
mkdir -p "$OUT_DIR"
# The vLLM env has the old TRL, which main's grpo.py targets (trust_remote_code lives on ModelConfig there)
git show main:src/phantom_reasoner/grpo.py > "$OUT_DIR/grpo_main.py"

for name in "$@"; do
    extra=()
    env_vars=()
    grpo=$REPO/src/phantom_reasoner/grpo.py
    env=$TF_ENV
    cfg=$R/config_pw_1gpu_smoke_cb.yaml
    case ${name%_n32} in
        vllm) env=$VLLM_ENV; cfg=$R/config_pw_1gpu_smoke.yaml; grpo=$OUT_DIR/grpo_main.py ;;
        cb) ;;
        cb_nograph) extra=(--transformers_continuous_batching_config '{"num_blocks": 256, "use_cuda_graph": false}') ;;
        cb_compile) extra=(--transformers_continuous_batching_config '{"num_blocks": 256, "default_compile_level": 1}')
            env_vars+=(DYNAMO_DISABLE_ATTN=1) ;;
        cb_kv[0-9]*) kv=${name%_n32}; kv=${kv#cb_kv}
            extra=(--transformers_continuous_batching_config "{\"num_blocks\": 256, \"kv_padding_interval_size\": $kv}") ;;
        generate) cfg=$R/config_pw_1gpu_smoke_generate.yaml ;;
        *) echo "Unknown variant: $name"; exit 1 ;;
    esac
    if [[ $name == *_n32 ]]; then
        extra+=(--gradient_accumulation_steps 8 --num_generations 8)
        # vLLM's backward OOMs from fragmentation at this size without expandable segments
        [[ $name == vllm_n32 ]] && env_vars+=(PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)
        # Padded generate() is very slow (and OOMs on 24GB): stop just after the profiling window
        [[ $name == generate_n32 ]] && extra+=(--max_steps $((${PROFILE_END:-4} + 1)))
    fi
    E=$CONDA_ENVS/$env
    profiler=()
    if $use_nsys; then
        profiler=($NSYS profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none
            --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true -o "$OUT_DIR/$name")
    fi
    env PATH="$E/bin:$PATH" CONDA_PREFIX="$E" GRPO_PY="$grpo" "${env_vars[@]}" "${profiler[@]}" \
        "$E/bin/accelerate" launch --num_processes 1 --config_file recipes/accelerate_configs/multi_gpu.yaml \
        benchmarks/bench_grpo.py --config "$cfg" --report_to none "${extra[@]}" > "$OUT_DIR/$name.log" 2>&1
    echo "$name exit=$?"
    if $use_nsys; then
        $NSYS export --type=sqlite --force-overwrite=true -o "$OUT_DIR/$name.sqlite" "$OUT_DIR/$name.nsys-rep" \
            >> "$OUT_DIR/$name.log" 2>&1
        echo "$name export exit=$?"
    fi
    sleep 5
done
