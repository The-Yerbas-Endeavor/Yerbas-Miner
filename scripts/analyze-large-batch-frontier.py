#!/usr/bin/env python3
"""Analyze raw-batch throughput/latency frontiers from cuda-batch-benchmark logs."""

from __future__ import annotations

import argparse
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

CASE_RE = re.compile(
    r"^### GPU (?P<gpu>\d+) \| CN triple (?P<triple>[^|]+?) \| pass=(?P<pass>\d+)\s*$"
)
REQUEST_RE = re.compile(
    r"^=== requested (?P<requested>\d+) \| actual (?P<actual>\d+) ===$"
)
RESULT_RE = re.compile(
    r"^GPU total: (?P<ms>[0-9.]+) ms \| wall: (?P<wall>[0-9.]+) ms \| throughput: (?P<hps>[0-9.]+) H/s$"
)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--plateau", type=float, default=0.99,
                    help="fraction of peak throughput considered equivalent (default: 0.99)")
    args = ap.parse_args()

    gpu = None
    triple = None
    pass_no = None
    requested = None
    actual = None
    samples: dict[tuple[int, str, int], list[tuple[float, float, int, int]]] = defaultdict(list)

    for raw in args.log.read_text(errors="replace").splitlines():
        line = raw.strip()
        m = CASE_RE.match(line)
        if m:
            gpu = int(m["gpu"])
            triple = m["triple"].strip()
            pass_no = int(m["pass"])
            requested = actual = None
            continue

        m = REQUEST_RE.match(line)
        if m and gpu is not None and triple is not None:
            requested = int(m["requested"])
            actual = int(m["actual"])
            continue

        m = RESULT_RE.match(line)
        if m and gpu is not None and triple is not None and actual is not None:
            ms = float(m["ms"])
            hps = float(m["hps"])
            samples[(gpu, triple, actual)].append((hps, ms, pass_no or 0, requested or actual))
            requested = actual = None

    if not samples:
        print("No frontier samples found.")
        return 2

    print("=== Yerbas large-batch throughput/latency frontier ===")
    print(f"Plateau threshold: {args.plateau*100.0:.2f}% of peak raw H/s")
    print()

    group_keys = sorted({(gpu, triple) for gpu, triple, _ in samples})
    recommendations: list[tuple[int, str, int, float, float, int, float]] = []

    for gpu, triple in group_keys:
        rows = []
        for (g, t, batch), vals in samples.items():
            if g != gpu or t != triple:
                continue
            hps_vals = [v[0] for v in vals]
            ms_vals = [v[1] for v in vals]
            rows.append((
                batch,
                statistics.median(hps_vals),
                statistics.median(ms_vals),
                len(vals),
                min(hps_vals),
                max(hps_vals),
            ))
        rows.sort()
        peak = max(r[1] for r in rows)
        threshold = peak * args.plateau
        plateau_rows = [r for r in rows if r[1] >= threshold]
        selected = min(plateau_rows, key=lambda r: r[0])
        peak_row = max(rows, key=lambda r: r[1])

        recommendations.append((
            gpu, triple, selected[0], selected[1], selected[2], peak_row[0], peak
        ))

        print(f"GPU {gpu} | {triple}")
        print("  batch   median H/s   median ms   vs peak   samples   range H/s")
        for batch, hps, ms, n, lo, hi in rows:
            rel = 100.0 * hps / peak if peak else 0.0
            mark = " <- plateau choice" if batch == selected[0] else ""
            if batch == peak_row[0]:
                mark += " <- peak"
            print(
                f"  {batch:5d}   {hps:10.2f}   {ms:9.2f}   {rel:7.2f}%"
                f"   {n:7d}   {lo:7.2f}-{hi:7.2f}{mark}"
            )
        saved = peak_row[2] - selected[2]
        latency_pct = 100.0 * saved / peak_row[2] if peak_row[2] else 0.0
        raw_loss = 100.0 * (1.0 - selected[1] / peak) if peak else 0.0
        print(
            f"  recommendation: batch={selected[0]} | raw loss={raw_loss:.2f}% "
            f"| scan latency={selected[2]:.2f} ms vs peak-batch {peak_row[0]} "
            f"{peak_row[2]:.2f} ms | latency reduction={latency_pct:.2f}%"
        )
        print()

    print("Cross-GPU/triple conservative ceiling:")
    chosen = max(r[2] for r in recommendations)
    print(f"  largest per-case plateau choice: {chosen}")
    print("  Keep a common production ceiling only if it remains inside the plateau")
    print("  on every target GPU/triple and improves live useful H/s.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
