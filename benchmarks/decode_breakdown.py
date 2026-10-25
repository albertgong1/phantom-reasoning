#!/usr/bin/env -S uv run python
"""Break down GPU time inside recorded generation steps of an nsys SQLite trace by kernel category.

Complements analyze_nsys.py: categories cover attention, GEMMs, torch.compile Triton kernels, PyTorch
elementwise and reduction kernels, sampling and copies. It also reports GPU busy time (merged across
streams) against wall time, and the top kernels by short name. Expects traces from run_benchmarks.sh /
bench_grpo.py ("gen step N" NVTX ranges).

Usage: python benchmarks/decode_breakdown.py runs/benchmarks/prof_n32/cb_n32.sqlite [more.sqlite ...]
"""

import bisect
import collections
import re
import sqlite3
import sys

CATS = [
    ("attn", re.compile(r"flash_fwd|flash::|reshape_and_cache|paged_attention", re.I)),
    (
        "sampling",
        re.compile(r"sort|topk|top_k|softmax|multinomial|cumsum|radix|argmax|exponential|gumbel|sampl", re.I),
    ),
    ("gemm", re.compile(r"gemm|gemv|cutlass|cublas|splitKreduce|sm80_xmma|ampere_", re.I)),
    ("triton", re.compile(r"^triton_", re.I)),
    ("copy/cat/index", re.compile(r"copy|CatArray|index|gather|scatter|fill|memset", re.I)),
    ("elementwise/reduce", re.compile(r"elementwise|reduce_kernel|vectorized|unrolled", re.I)),
]


def cat(name):
    for c, rx in CATS:
        if rx.search(name):
            return c
    return "other"


def main(path):
    con = sqlite3.connect(path)
    strings = dict(con.execute("SELECT id, value FROM StringIds"))
    cols = {r[1] for r in con.execute("PRAGMA table_info(NVTX_EVENTS)")}
    tid = "textId" if "textId" in cols else "NULL"
    ranges = sorted(
        (s, e, (t or strings.get(i, "")))
        for s, e, t, i in con.execute(
            f"SELECT start, end, text, {tid} FROM NVTX_EVENTS WHERE end IS NOT NULL"
        )
        if (t or strings.get(i, "")).startswith("gen step")
    )
    starts = [r[0] for r in ranges]
    per = collections.defaultdict(lambda: [0, 0.0])
    names = collections.defaultdict(lambda: [0, 0.0])
    busy_intervals = []
    for s, e, n in con.execute("SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        i = bisect.bisect_right(starts, s) - 1
        if i < 0 or s > ranges[i][1]:
            continue
        name = strings.get(n, str(n))
        c = cat(name)
        per[c][0] += 1
        per[c][1] += (e - s) / 1e6
        names[name][0] += 1
        names[name][1] += (e - s) / 1e6
        busy_intervals.append((s, e))
    # merged busy time (kernels can overlap across streams)
    busy_intervals.sort()
    merged, cur_s, cur_e = 0, None, None
    for s, e in busy_intervals:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                merged += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        merged += cur_e - cur_s
    wall = sum(e - s for s, e, _ in ranges) / 1e6
    total = sum(v[1] for v in per.values())
    nk = sum(v[0] for v in per.values())
    print(f"\n===== {path} =====  ranges: {[r[2] for r in ranges]}")
    print(
        f"wall {wall:.0f} ms | GPU busy (merged) {merged / 1e6:.0f} ms ({100 * merged / 1e6 / wall:.0f}%) | "
        f"kernel sum {total:.0f} ms | kernels {nk}"
    )
    for c, (n, ms) in sorted(per.items(), key=lambda x: -x[1][1]):
        print(
            f"  {c:20s} {ms:8.0f} ms  {100 * ms / total:5.1f}%  {n:8d} kernels  "
            f"{1000 * ms / max(n, 1):7.1f} us/kernel"
        )
    print("  top kernels:")
    for name, (n, ms) in sorted(names.items(), key=lambda x: -x[1][1])[:15]:
        print(f"    {ms:8.0f} ms {n:7d}x {1000 * ms / n:7.1f} us [{cat(name)}] {name[:90]}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
