#!/usr/bin/env python3
"""Compare full-pipeline batch-7168 throughput with setup threads 32 vs 128."""

from __future__ import annotations

import argparse
import re
import statistics
from collections import defaultdict
from pathlib import Path

CASE_RE = re.compile(
    r"^### GPU (?P<gpu>\d+) \| triple (?P<triple>[^|]+?) "
    r"\| repeat=(?P<repeat>\d+) \| setup=(?P<setup>32|128)\s*$"
)
RESULT_RE = re.compile(
    r"^GPU total: (?P<ms>[0-9.]+) ms \| wall: (?P<wall>[0-9.]+) ms "
    r"\| throughput: (?P<hps>[0-9.]+) H/s$"
)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    args = ap.parse_args()

    current = None
    samples = defaultdict(list)

    for raw in args.log.read_text(errors="replace").splitlines():
        line = raw.strip()
        m = CASE_RE.match(line)
        if m:
            current = (
                int(m["gpu"]),
                m["triple"].strip(),
                int(m["repeat"]),
                int(m["setup"]),
            )
            continue

        m = RESULT_RE.match(line)
        if m and current is not None:
            gpu, triple, repeat, setup = current
            samples[(gpu, triple, setup)].append(
                (repeat, float(m["hps"]), float(m["ms"]), float(m["wall"]))
            )
            current = None

    if not samples:
        print("No A/B samples found.")
        return 2

    print("=== Yerbas batch-7168 setup geometry whole-pipeline A/B ===")
    print()

    overall = {32: [], 128: []}
    all_positive = True

    groups = sorted({(gpu, triple) for gpu, triple, _ in samples})
    for gpu, triple in groups:
        a = samples.get((gpu, triple, 32), [])
        b = samples.get((gpu, triple, 128), [])
        if not a or not b:
            print(f"GPU {gpu} | {triple}: incomplete A/B")
            all_positive = False
            continue

        h32 = [x[1] for x in a]
        h128 = [x[1] for x in b]
        m32 = statistics.median(h32)
        m128 = statistics.median(h128)
        delta = 100.0 * (m32 / m128 - 1.0) if m128 else 0.0

        paired = []
        by_repeat_32 = {r: h for r, h, *_ in a}
        by_repeat_128 = {r: h for r, h, *_ in b}
        for repeat in sorted(set(by_repeat_32) & set(by_repeat_128)):
            base = by_repeat_128[repeat]
            cand = by_repeat_32[repeat]
            paired.append(100.0 * (cand / base - 1.0) if base else 0.0)

        positive_pairs = sum(1 for x in paired if x > 0.0)
        if delta <= 0.0 or positive_pairs < max(1, len(paired) - 1):
            all_positive = False

        overall[32].extend(h32)
        overall[128].extend(h128)

        pair_text = ", ".join(f"{x:+.3f}%" for x in paired)
        print(
            f"GPU {gpu} | {triple:26s} | "
            f"setup32={m32:8.2f} H/s setup128={m128:8.2f} H/s "
            f"delta={delta:+.3f}% | paired=[{pair_text}]"
        )

    print()
    if overall[32] and overall[128]:
        o32 = statistics.median(overall[32])
        o128 = statistics.median(overall[128])
        odelta = 100.0 * (o32 / o128 - 1.0) if o128 else 0.0
        print(
            f"Overall sample median: setup32={o32:.2f} H/s "
            f"setup128={o128:.2f} H/s delta={odelta:+.3f}%"
        )
    else:
        odelta = 0.0

    print()
    print("Production gate:")
    if all_positive and odelta > 0.0:
        print("  PASS candidate: setup=32 is directionally positive across both GPUs/triples.")
        print("  Validate in live production before merging.")
        return 0

    print("  HOLD: whole-pipeline benefit is not sufficiently consistent.")
    print("  Keep setup=128 fallback despite phase-local microbenchmark gains.")
    return 3

if __name__ == "__main__":
    raise SystemExit(main())
