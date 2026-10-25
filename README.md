# Phantom Reasoning

**[Learning from Synthetic Data Improves Multi-hop Reasoning](https://arxiv.org/abs/2603.02091)** — ICLR 2026

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2603.02091)
[![Dataset](https://img.shields.io/badge/Datasets-Huggingface-yellow)](https://huggingface.co/datasets/kilian-group/phantom-reasoning)
[![Conference](https://img.shields.io/badge/ICLR-2026-blue)](https://iclr.cc/Conferences/2026)

We RL fine-tune on rule-generated synthetic data (PhantomWiki, GSM-Infinite, ReasoningGym) and show transfer to real-world multi-hop reasoning  (HotpotQA, 2Wiki, Musique, CofCA, SynthWorlds-RM).

<p align="center">
  <img src="assets/motivation.jpg" alt="Motivation" width="100%"/>
</p>

## Quick start

The fastest path from a fresh clone to a trained-and-evaluated model on **a single GPU**. See [Scaling up](#scaling-up) for the multi-GPU and cluster recipes used in the paper.

**1. Install** (Python 3.12+, [uv](https://github.com/astral-sh/uv), [SWI-Prolog](https://www.swi-prolog.org/)):

```bash
conda create -n phantom-reasoning python=3.12
conda activate phantom-reasoning
conda install conda-forge::swi-prolog   # PhantomWiki uses Prolog to generate/solve questions
pip install uv

git clone git@github.com:kilian-group/phantom-reasoning.git
cd phantom-reasoning
uv pip install -e ".[dev]"
# Prebuilt wheel matching torch 2.9 (pinned by vLLM); `uv pip install flash-attn` builds from source and can fail
uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
```

**2. Download the data** (PhantomWiki + all eval sets, into `data/`):

```bash
./scripts/download_data.sh data
```

**3. Smoke-test the training loop** (a few minutes, Qwen3-0.6B, 10 steps, no checkpoints, tested on 1 A6000 with 48GB memory):

```bash
accelerate launch --num_processes 1 --config_file recipes/accelerate_configs/multi_gpu.yaml \
    src/phantom_reasoner/grpo.py \
    --config recipes/Qwen/Qwen3-0.6B/grpo/config_pw_1gpu_smoke.yaml \
    --report_to none
```

**4. Train for real** on one GPU (Qwen3-0.6B on PhantomWiki, tested on 1 H100/B200):

```bash
accelerate launch --num_processes 1 --config_file recipes/accelerate_configs/multi_gpu.yaml \
    src/phantom_reasoner/grpo.py \
    --config recipes/Qwen/Qwen3-0.6B/grpo/config_pw_1gpu.yaml \
    --report_to none
```

Checkpoints are written to `$RUN_BASE_DIR/runs/<dataset>/<model>/grpo/$USER/<MMDD>__<flags>` (`RUN_BASE_DIR` defaults to `.`, so e.g. `./runs/data/wiki-v1-easy-depth_20_size_25/Qwen/Qwen3-0.6B/grpo/$USER/...`). Watch the `reward` curve climb in the logs.

> **Logging:** training logs to [Weights & Biases](https://wandb.ai) by default. Either run `wandb login` first, or pass `--report_to none` (as above) to disable it.

**5. Evaluate** the transfer to a real-world dataset (HotpotQA), for both the base model and your checkpoint:

```bash
# Base model baseline
MODEL_NAMES="Qwen/Qwen3-0.6B" bash scripts/eval/wiki_eval_grpo.sh out__eval=base hp500 minidev

# Your trained checkpoint (point MODEL_NAMES at the checkpoint directory from step 4)
MODEL_NAMES="./runs/.../checkpoint-XXX" bash scripts/eval/wiki_eval_grpo.sh out__eval=mytrain hp500 minidev
```

Per-sample predictions land in the `out__eval=*` directories; aggregate F1 is printed and saved by `examples/wiki/format_split_accuracy.py`. Swap `hp500` for `2wiki500`, `msq500`, `cofca500`, or `synthrm500`.

## Scaling up

<details>
<summary><strong>Multi-GPU training (the paper recipes)</strong></summary>

Recipes live under `recipes/<org>/<model>/grpo/`. The `*_4gpu` / `*_2gpu` suffix indicates the GPU count they are tuned for - you can play around with `vllm_gpu_memory_utilization`, batch sizes, and number of generations if you go CUDA OOM. Launch directly with `accelerate` — `--num_processes` should match your GPU count:

```bash
# 4-GPU GRPO: Qwen3-1.7B on PhantomWiki
accelerate launch --num_processes 4 --config_file recipes/accelerate_configs/zero1.yaml \
    src/phantom_reasoner/grpo.py \
    --config recipes/Qwen/Qwen3-1.7B/grpo/config_pw_4gpu.yaml
```

Available models: `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `microsoft/Phi-4-mini-reasoning`. Each has recipes for PhantomWiki (`config_pw_*`), GSM-Infinite (`config_gsminfinite_*`), ReasoningGym (`config_rg-*`), and the real-world datasets (`config_hp_*`, `config_2wiki_*`, `config_msq_*`).

Accelerate configs in `recipes/accelerate_configs/`: `multi_gpu.yaml` (DDP), `zero1.yaml`/`zero2.yaml`/`zero3.yaml` (DeepSpeed ZeRO stages 1/2/3).

</details>

<details>
<summary><strong>Running on a SLURM cluster</strong></summary>

Generate a submission script for your cluster (supported: `aida`, `anvil`, `empire`, `unicorn`; omit the name for a generic SLURM setup):

```bash
./scripts/create_train_grpo__vllm_colocate.sh [cluster_name]
```

It wraps the same `accelerate launch` command and auto-detects the GPU count from `nvidia-smi`. Submit with `sbatch`:

```bash
sbatch scripts/train_grpo__vllm_colocate.sub \
    recipes/accelerate_configs/zero1.yaml \
    recipes/Qwen/Qwen3-1.7B/grpo/config_pw_4gpu.yaml
```

Cluster-specific setup (data paths, modules, environment variables): [docs/README_anvil.md](docs/README_anvil.md), [docs/README_unicorn.md](docs/README_unicorn.md).

To persist environment variables (`RUN_BASE_DIR`, `HF_HOME`, W&B, SLURM email) in your conda env:

```bash
./scripts/setup_conda_env_vars.sh phantom-reasoning [cluster_name]
```

</details>

<details>
<summary><strong>Generating synthetic data from scratch</strong></summary>

The released data on [HuggingFace](https://huggingface.co/datasets/kilian-group/phantom-reasoning) is enough to reproduce the paper. To regenerate:

- **PhantomWiki:** install the [phantom-wiki](https://github.com/kilian-group/phantom-wiki) package, then generate one universe (a single family tree of up to 25 people) per seed — seeds 1–10 are reserved for evaluation, 11+ for training:
  ```bash
  # depth_20_size_25 with --easy-mode — the split the recipes train on
  for seed in $(seq 1 41); do
    python -m phantom_wiki \
      -od data/wiki-v1-easy-depth_20_size_25/depth_20_size_25_seed_${seed} \
      --seed ${seed} \
      --question-depth 20 --max-family-tree-depth 20 \
      --num-family-trees 1 --max-family-tree-size 25 \
      --article-format json --question-format json \
      --easy-mode
  done
  ```
  For the `depth_30_size_25` reasoning-evolution eval split, change `20 → 30` in both the depth flags and the output path.
- **GSM-Infinite:** see [gsm_realistic/README.md](gsm_realistic/README.md).
- **ReasoningGym:**
  ```bash
  python scripts/generate_reasoning_gym_data.py --dataset $task --size 12500 --train_frac 0.8 -od data/rg-family_relationships
  python scripts/generate_reasoning_gym_data.py --dataset $task --size 12500 --train_frac 0.8 -od data/rg-knights_knaves
  ```

</details>

## Evaluation

The quick start covers real-world transfer eval. For the full evaluation suite — PhantomWiki and GSM-Infinite eval, evaluating all intermediate checkpoints, reproducing the paper's reasoning-evolution and transfer plots, and our released checkpoints — see [docs/EVALUATION.md](docs/EVALUATION.md). Released checkpoint paths are in `scripts/final_plots/final_ckpts.yaml`.

## Citation

```bibtex
@inproceedings{kabra2026learning,
  title={{Learning from Synthetic Data Improves Multi-hop Reasoning}},
  author={Kabra, Anmol and Gong, Albert and Yin, Yilun and Stankevi{\v{c}}i{\=u}t{\.e}, Kamil{\.e} and Go, Dongyoung and Luo, Katie Z and Lee, Johann and Gomes, Carla P and Weinberger, Kilian Q},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://arxiv.org/abs/2603.02091}
}
```
