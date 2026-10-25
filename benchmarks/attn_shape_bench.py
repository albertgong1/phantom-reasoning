#!/usr/bin/env -S uv run python
"""Time FlashAttention calls at the shapes transformers continuous batching produces for Qwen3-0.6B.

- Prefill (flash_attn_varlen_func): does CUDA-graph padding (max_seqlen_q = whole padded batch, extra
  empty sequences) slow prefill down, and what do warmup-shaped batches (8,192 queries against ~65k KV)
  cost?
- Decode (flash_attn_with_kvcache): does appending the new K/V inside the kernel (continuous batching)
  cost more than a read-only call over a pre-written cache (vLLM), and does num_splits matter?

Usage: python benchmarks/attn_shape_bench.py
"""

import torch
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

H, HKV, D = 16, 8, 128
DEV, DT = "cuda", torch.bfloat16


def time_ms(f, iters=20):
    for _ in range(3):
        f()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        f()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def prefill(seqlens_q, seqlens_k, max_q, max_k, n_pad_seqs=0):
    tq, tk = sum(seqlens_q), sum(seqlens_k)
    q = torch.randn(tq, H, D, device=DEV, dtype=DT)
    k = torch.randn(tk, HKV, D, device=DEV, dtype=DT)
    v = torch.randn_like(k)
    cu_q = torch.tensor([0, *torch.tensor(seqlens_q).cumsum(0).tolist()] + [tq] * n_pad_seqs, device=DEV)
    cu_k = torch.tensor([0, *torch.tensor(seqlens_k).cumsum(0).tolist()] + [tk] * n_pad_seqs, device=DEV)
    cu_q, cu_k = cu_q.to(torch.int32), cu_k.to(torch.int32)
    return time_ms(lambda: flash_attn_varlen_func(q, k, v, cu_q, cu_k, max_q, max_k, causal=True))


def decode(batch=4, ctx=5600, page=256):
    nblk = (ctx + page) // page + 1
    k_cache = torch.randn(batch * nblk, page, HKV, D, device=DEV, dtype=DT)
    v_cache = torch.randn_like(k_cache)
    table = torch.arange(batch * nblk, device=DEV, dtype=torch.int32).view(batch, nblk)
    lens = torch.full((batch,), ctx, device=DEV, dtype=torch.int32)
    q = torch.randn(batch, 1, H, D, device=DEV, dtype=DT)
    k_new, v_new = torch.randn(batch, 1, HKV, D, device=DEV, dtype=DT), torch.randn(
        batch, 1, HKV, D, device=DEV, dtype=DT
    )
    append = time_ms(
        lambda: flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k_new, v=v_new, cache_seqlens=lens, block_table=table, causal=True
        ),
        200,
    )
    print(f"  decode, append K/V in kernel (CB):             {1000 * append:7.1f} us")
    for ns in (0, 1, 2, 4, 8, 16):
        t = time_ms(
            lambda: flash_attn_with_kvcache(
                q, k_cache, v_cache, cache_seqlens=lens + 1, block_table=table, causal=True, num_splits=ns
            ),
            200,
        )
        print(f"  decode, read-only (vLLM-style), num_splits={ns:2d}: {1000 * t:7.1f} us")


if __name__ == "__main__":
    print("Prefill (flash_attn_varlen_func):")
    cases = {
        "2 prompts (5114 + 3078), exact max_q": ([5114, 3078], [5114, 3078], 5114, 5114, 0),
        "same, max_q = max_k = 8192 (CB padding)": ([5114, 3078], [5114, 3078], 8192, 8192, 0),
        "same + 2 empty padded sequences": ([5114, 3078], [5114, 3078], 8192, 8192, 2),
        "1 prompt of 4096 (vLLM chunk)": ([4096], [4096], 4096, 4096, 0),
        "warmup shape: q=8192, k=65536": ([8192], [65536], 8192, 65536, 3),
        "q=8192 with a 16k-token cached prefix": ([8192], [24576], 8192, 24576, 3),
        "mixed: q=[1, 2036, 250, 250], k=~5.1k each": (
            [1, 2036, 250, 250],
            [5115, 5114, 5114, 5114],
            8192,
            8192,
            0,
        ),
    }
    for name, args in cases.items():
        print(f"  {name:48s} {prefill(*args):7.2f} ms")
    print("Decode (flash_attn_with_kvcache, 4 sequences x 5.6k context, 256-token pages):")
    decode()
