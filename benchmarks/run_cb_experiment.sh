#!/usr/bin/env bash
# Run one transformers continuous batching (CB) GRPO experiment without nsys, with per-step tracing (CB_TRACE=1:
# generation time, prompt tokens, prefill tokens actually scheduled and forward passes, as "[bench]" log lines).
#
# The CB config JSON replaces the recipe's whole transformers_continuous_batching_config dict. Extra environment
# variables (e.g. CB_PERSISTENT=1 for bench_grpo.py's persistent manager) go before an optional "--"; extra GRPO
# arguments go after it.
#
# Usage: benchmarks/run_cb_experiment.sh <name> '<cb_config_json>' [VAR=VALUE ...] [-- extra grpo args]
#   e.g. benchmarks/run_cb_experiment.sh n32_blocks512 \
#            '{"num_blocks": 512, "default_compile_level": 1, "use_async_batching": false, "max_batch_tokens": 2048}' \
#            -- --gradient_accumulation_steps 8 --num_generations 8
# Env: TF_ENV (conda env name, default phantom-reasoning-tf), CONDA_ENVS (default ~/.conda/envs),
#      OUT_DIR (default runs/benchmarks/hyp), CONFIG (default the Qwen3-0.6B CB smoke recipe)

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <name> '<cb_config_json>' [VAR=VALUE ...] [-- extra grpo args]"
    exit 1
fi
name=$1
json=$2
shift 2
env_vars=()
while [[ $# -gt 0 && $1 != -- ]]; do
    env_vars+=("$1")
    shift
done
[[ $1 == -- ]] && shift
cmd_args=("$@")

REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
E=${CONDA_ENVS:-$HOME/.conda/envs}/${TF_ENV:-phantom-reasoning-tf}
OUT_DIR=${OUT_DIR:-$REPO/runs/benchmarks/hyp}
CONFIG=${CONFIG:-recipes/Qwen/Qwen3-0.6B/grpo/config_pw_1gpu_smoke_cb.yaml}
cd "$REPO" || exit 1
mkdir -p "$OUT_DIR"
env PATH="$E/bin:$PATH" CONDA_PREFIX="$E" GRPO_PY="$REPO/src/phantom_reasoner/grpo.py" CB_TRACE=1 "${env_vars[@]}" \
    "$E/bin/accelerate" launch --num_processes 1 --config_file recipes/accelerate_configs/multi_gpu.yaml \
    benchmarks/bench_grpo.py --config "$CONFIG" --report_to none \
    --transformers_continuous_batching_config "$json" "${cmd_args[@]}" > "$OUT_DIR/$name.log" 2>&1
echo "$name exit=$?"
