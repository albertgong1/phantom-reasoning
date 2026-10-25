#!/usr/bin/env -S uv run python
"""Summarize Nsight Systems SQLite exports: GPU kernel time in generation vs training, and attention kernels.

A kernel counts as generation if its GPU start time falls inside a "gen step N" NVTX range (pushed by
bench_grpo.py on the main thread; continuous batching launches kernels from a background thread, so ranges are
matched by time, not thread). Requires the trace to be taken with `--cuda-graph-trace=node` for kernels inside
CUDA graph replays to be visible.

Usage: python benchmarks/analyze_nsys.py runs/benchmarks/vllm.sqlite runs/benchmarks/cb.sqlite ...
"""

import bisect
import collections
import re
import sqlite3
import sys

ATTN_FWD = re.compile(
    r"flash_fwd|fmha.*(fwd|forward)|flash_attn|paged_attention|attention_kernel|attn_fwd", re.I
)
ATTN_BWD = re.compile(r"flash_bwd|fmha.*(bwd|backward)|attn_bwd", re.I)


def kernel_category(name):
    if ATTN_BWD.search(name):
        return "attn_bwd"
    if ATTN_FWD.search(name):
        return "attn_fwd"
    return "other"


def load(path):
    """Load a trace.

    Returns:
        (ranges, kernels): sorted (start_ns, end_ns, step) generation ranges, and
        (start_ns, end_ns, name) kernels.
    """
    con = sqlite3.connect(path)
    strings = dict(con.execute("SELECT id, value FROM StringIds"))
    cols = {r[1] for r in con.execute("PRAGMA table_info(NVTX_EVENTS)")}
    text_id_col = "textId" if "textId" in cols else "NULL"
    ranges = []
    for start, end, text, text_id in con.execute(
        f"SELECT start, end, text, {text_id_col} FROM NVTX_EVENTS WHERE end IS NOT NULL"
    ):
        label = text if text else strings.get(text_id, "")
        if label.startswith("gen step"):
            ranges.append((start, end, int(label.split()[-1])))
    ranges.sort()
    kernels = [
        (start, end, strings.get(name_id, str(name_id)))
        for start, end, name_id in con.execute(
            "SELECT start, end, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL"
        )
    ]
    return ranges, kernels


def summarize(path):
    ranges, kernels = load(path)
    starts = [r[0] for r in ranges]
    busy = collections.defaultdict(float)  # (phase, category) -> ms
    by_name = collections.defaultdict(lambda: [0, 0.0])  # (phase, name) -> [count, ms]
    gen_step_ms = collections.defaultdict(lambda: collections.defaultdict(float))
    for k_start, k_end, name in kernels:
        i = bisect.bisect_right(starts, k_start) - 1
        in_gen = i >= 0 and k_start <= ranges[i][1]
        phase = "gen" if in_gen else "train"
        ms = (k_end - k_start) / 1e6
        cat = kernel_category(name)
        busy[(phase, cat)] += ms
        by_name[(phase, name)][0] += 1
        by_name[(phase, name)][1] += ms
        if in_gen:
            gen_step_ms[ranges[i][2]][cat] += ms

    print(f"\n===== {path} =====")
    print("Per-step generation (wall = NVTX range; kernel times = summed GPU kernel durations):")
    for start, end, step in ranges:
        d = gen_step_ms[step]
        total = sum(d.values())
        attn_pct = 100 * d["attn_fwd"] / max(total, 1e-9)
        print(
            f"  step {step}: wall {(end - start) / 1e6:8.1f} ms | kernels {total:8.1f} ms | "
            f"attn fwd {d['attn_fwd']:7.1f} ms ({attn_pct:4.1f}% of kernel time)"
        )
    for phase in ("gen", "train"):
        total = sum(v for (p, _), v in busy.items() if p == phase)
        parts = ", ".join(f"{c} {busy[(phase, c)]:.1f}" for c in ("attn_fwd", "attn_bwd", "other"))
        print(f"{phase:5s} kernel total {total:9.1f} ms  ({parts})")
        top = sorted(((v[1], v[0], n) for (p, n), v in by_name.items() if p == phase), reverse=True)[:12]
        for ms, count, name in top:
            print(f"    {ms:9.1f} ms {count:7d}x [{kernel_category(name):8s}] {name[:110]}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarize(p)
