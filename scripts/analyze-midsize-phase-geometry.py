#!/usr/bin/env python3
"""Summarize 5376/7168 CryptoNight setup/final geometry measurements."""

from __future__ import annotations

import argparse
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

GEOM_RE = re.compile(
    r"^\[CUDA CN phase geometry\] GPU (?P<gpu>\d+) \| "
    r"(?P<variant>CN-[^|]+?) \| phase=(?P<phase>setup|final) "
    r"\| batch=(?P<batch>\d+) "
    r"\| 32=(?P<t32>[0-9.]+) ms "
    r"\| 64=(?P<t64>[0-9.]+) ms "
    r"\| 96=(?P<t96>[0-9.]+) ms "
    r"\| 128=(?P<t128>[0-9.]+) ms "
    r"\| threshold=1% \| production=(?P<production>\d+)\s*$"
)

THREADS = (32, 64, 96, 128)


def choose(medians: dict[int, float]) -> int:
    selected = 128
    best = medians[128]
    for threads in (32, 64, 96):
        if medians[threads] < medians[128] * 0.99 and medians[threads] < best:
            best = medians[threads]
            selected = threads
    return selected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    args = ap.parse_args()

    samples: dict[tuple[int, int, str, str], list[dict[int, float]]] = defaultdict(list)
    choices: dict[tuple[int, int, str, str], list[int]] = defaultdict(list)

    for raw in args.log.read_text(errors="replace").splitlines():
        m = GEOM_RE.match(raw.strip())
        if not m:
            continue
        key = (
            int(m["gpu"]),
            int(m["batch"]),
            m["variant"].strip(),
            m["phase"],
        )
        samples[key].append({
            32: float(m["t32"]),
            64: float(m["t64"]),
            96: float(m["t96"]),
            128: float(m["t128"]),
        })
        choices[key].append(int(m["production"]))

    if not samples:
        print("No [CUDA CN phase geometry] samples found.")
        return 2

    print("=== Yerbas mid-size CN phase geometry ===")
    print("Aggregate rule matches production selector: candidate must beat 128 by >=1%.")
    print()

    promoted = []
    for key in sorted(samples):
        gpu, batch, variant, phase = key
        rows = samples[key]
        medians = {
            t: statistics.median(row[t] for row in rows)
            for t in THREADS
        }
        selected = choose(medians)
        observed = Counter(choices[key])
        base = medians[128]
        gain = 100.0 * (1.0 - medians[selected] / base) if base else 0.0

        choice_text = ", ".join(f"{t}:{n}" for t, n in sorted(observed.items()))
        print(
            f"GPU {gpu} batch={batch:4d} {variant:13s} phase={phase:5s} "
            f"samples={len(rows)} | "
            f"32={medians[32]:8.3f} 64={medians[64]:8.3f} "
            f"96={medians[96]:8.3f} 128={medians[128]:8.3f} ms | "
            f"aggregate={selected:3d} gain_vs_128={gain:6.2f}% | "
            f"per-run=[{choice_text}]"
        )
        if selected != 128:
            promoted.append((gpu, batch, variant, phase, selected, gain))

    print()
    print("Promotion candidates:")
    if not promoted:
        print("  none; keep conservative 128-thread fallback at 5376/7168")
    else:
        for gpu, batch, variant, phase, selected, gain in promoted:
            print(
                f"  GPU {gpu} batch={batch} {variant} {phase}: "
                f"{selected} threads ({gain:.2f}% faster than 128 median)"
            )

    print()
    print("Production recommendation gate:")
    print("  Promote only choices that repeat on both passes and are directionally")
    print("  consistent across both GTX 1080 Ti cards at the same batch/variant/phase.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
