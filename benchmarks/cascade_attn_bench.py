#!/usr/bin/env -S uv run python
"""Compare plain paged decode attention (transformers continuous batching) with vLLM-style cascade
attention.

Uses flash-attn 2.8.3 kernels with Qwen3-0.6B attention shapes (16 query heads, 8 KV heads, head dim 128,
bf16) and 256-token pages. Sequences come in prompt groups. Each sequence's block table points at COMMON
tokens shared by all sequences, then GROUP tokens shared within its prompt group, then GEN generated
tokens unique to the sequence. The defaults match the 32-completion GRPO smoke run (4 prompts x 8
completions, ~4.86k shared tokens).

- plain: one flash_attn_with_kvcache call over each sequence's full block table (what continuous batching
  does)
- cascade: flash_attn_varlen_func attending all queries to the shared pages once, flash_attn_with_kvcache
  over each sequence's remaining pages, and a log-sum-exp merge in PyTorch (a reimplementation of vLLM's
  approach)

Usage: python benchmarks/cascade_attn_bench.py [--common 4864] [--group 256] [--groups 4] [--per-group 8]
"""

import argparse

import torch
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

H, HKV, D, PAGE = 16, 8, 128, 256
DEV, DT = "cuda", torch.bfloat16


def time_us(f, iters=200):
    for _ in range(10):
        f()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        f()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000


def bench(common, group, gen, groups, per_group):
    b = groups * per_group
    nc, ng, nu = common // PAGE, group // PAGE, gen // PAGE + 1
    k_cache = torch.randn(nc + groups * ng + b * nu, PAGE, HKV, D, device=DEV, dtype=DT)
    v_cache = torch.randn_like(k_cache)
    common_pages = torch.arange(nc, device=DEV, dtype=torch.int32)
    group_pages = (nc + torch.arange(groups * ng, device=DEV, dtype=torch.int32)).view(groups, ng)
    own_pages = (nc + groups * ng + torch.arange(b * nu, device=DEV, dtype=torch.int32)).view(b, nu)
    group_of = torch.arange(b, device=DEV) // per_group
    full_table = torch.cat([common_pages.expand(b, -1), group_pages[group_of], own_pages], 1).contiguous()
    suffix_table = torch.cat([group_pages[group_of], own_pages], 1).contiguous()
    lens = torch.full((b,), common + group + gen, device=DEV, dtype=torch.int32)
    q = torch.randn(b, 1, H, D, device=DEV, dtype=DT)
    k_pre = k_cache[:nc].reshape(-1, HKV, D)
    v_pre = v_cache[:nc].reshape(-1, HKV, D)
    cu_q = torch.tensor([0, b], device=DEV, dtype=torch.int32)
    cu_k = torch.tensor([0, common], device=DEV, dtype=torch.int32)

    def plain(num_splits=0):
        return flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=lens,
            block_table=full_table,
            causal=True,
            num_splits=num_splits,
        )

    def prefix():
        return flash_attn_varlen_func(
            q.view(b, H, D), k_pre, v_pre, cu_q, cu_k, b, common, causal=False, return_attn_probs=True
        )

    def suffix():
        return flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=lens - common,
            block_table=suffix_table,
            causal=True,
            return_softmax_lse=True,
        )

    def cascade():
        o1, lse1, _ = prefix()
        o2, lse2 = suffix()
        lse1, lse2 = lse1.view(H, b).t(), lse2.view(
            b, H
        )  # varlen lse is (H, total_q); kvcache lse is (B, H, 1)
        m = torch.maximum(lse1, lse2)
        w1, w2 = torch.exp(lse1 - m), torch.exp(lse2 - m)
        return (o1.view(b, H, D) * w1[..., None] + o2.view(b, H, D) * w2[..., None]) / (w1 + w2)[..., None]

    err = (plain().view(b, H, D).float() - cascade().float()).abs().max().item()
    splits = {ns: time_us(lambda: plain(ns)) for ns in (0, 1, 2, 4, 8)}
    t_pre, t_suf, t_cascade = time_us(prefix), time_us(suffix), time_us(cascade)
    print(
        f"gen={gen:5d} ctx={common + group + gen}: plain {splits[0]:.0f} us "
        f"(num_splits 1/2/4/8: {' / '.join(f'{splits[n]:.0f}' for n in (1, 2, 4, 8))}) | cascade prefix "
        f"{t_pre:.0f} + suffix {t_suf:.0f} = {t_pre + t_suf:.0f} us ({t_cascade:.0f} incl. merge, "
        f"{splits[0] / (t_pre + t_suf):.2f}x) | max abs diff {err:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--common", type=int, default=4864, help="tokens shared by all sequences (page aligned)"
    )
    parser.add_argument(
        "--group", type=int, default=256, help="tokens shared within a prompt group (page aligned)"
    )
    parser.add_argument("--groups", type=int, default=4, help="number of prompts")
    parser.add_argument("--per-group", type=int, default=8, help="completions per prompt")
    args = parser.parse_args()
    for gen_tokens in (256, 512, 1024):
        bench(args.common, args.group, gen_tokens, args.groups, args.per_group)
