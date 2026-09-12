#!/usr/bin/env python3
"""Aggregate Yerbas Miner rotation telemetry by ordered CryptoNight triple.

Usage:
    python3 scripts/analyze-rotation-perf.py logs/rotation-perf-*.log

The miner already emits both:
    [GhostRider] rotation=<fingerprint> | CN=<ordered triple>
    [rotation perf] fingerprint=<fingerprint> | ...

This tool joins those records and computes hash-weighted throughput across all
fingerprints that share the same ordered CN triple. It intentionally performs
no hardware-specific classification or tuning.
"""

from __future__ import annotations

import argparse
import glob
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROTATION_RE = re.compile(
    r"\[GhostRider\]\s+rotation=([0-9a-fA-F]{16})\s+\|\s+CN=([^\r\n]+)"
)
PERF_RE = re.compile(
    r"\[rotation perf\]\s+fingerprint=([0-9a-fA-F]{16})"
    r"\s+\|\s+sample=(\d+)"
    r"\s+\|\s+duration=([0-9.]+)s"
    r"\s+\|\s+hashes=(\d+)"
    r"\s+\|\s+total=([0-9.]+)\s+H/s"
    r"\s+\|\s+cpu=([0-9.]+)\s+H/s"
    r"\s+\|\s+gpu=([0-9.]+)\s+H/s"
)


@dataclass
class Aggregate:
    samples: int = 0
    seconds: float = 0.0
    hashes: int = 0
    cpu_hashes: float = 0.0
    gpu_hashes: float = 0.0
    min_hps: float = float("inf")
    max_hps: float = 0.0

    def add(self, seconds: float, hashes: int, total_hps: float, cpu_hps: float, gpu_hps: float) -> None:
        self.samples += 1
        self.seconds += seconds
        self.hashes += hashes
        self.cpu_hashes += cpu_hps * seconds
        self.gpu_hashes += gpu_hps * seconds
        self.min_hps = min(self.min_hps, total_hps)
        self.max_hps = max(self.max_hps, total_hps)

    @property
    def hps(self) -> float:
        return self.hashes / self.seconds if self.seconds > 0.0 else 0.0

    @property
    def cpu_hps(self) -> float:
        return self.cpu_hashes / self.seconds if self.seconds > 0.0 else 0.0

    @property
    def gpu_hps(self) -> float:
        return self.gpu_hashes / self.seconds if self.seconds > 0.0 else 0.0


def expand_inputs(items: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for item in items:
        matches = [Path(p) for p in glob.glob(item)]
        if not matches:
            matches = [Path(item)]
        for path in matches:
            path = path.expanduser()
            if path.is_file() and path not in seen:
                seen.add(path)
                files.append(path)
    return files


def parse_files(paths: Iterable[Path]) -> tuple[dict[str, Aggregate], int, int]:
    by_order: dict[str, Aggregate] = defaultdict(Aggregate)
    fingerprint_order: dict[str, str] = {}
    perf_records = 0
    unmatched = 0

    for path in paths:
        # Fingerprints can recur across files, but the schedule mapping is stable.
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = raw.replace("\x1b", "")
                rotation = ROTATION_RE.search(line)
                if rotation:
                    fingerprint_order[rotation.group(1).lower()] = rotation.group(2).strip()
                    continue

                perf = PERF_RE.search(line)
                if not perf:
                    continue

                perf_records += 1
                fingerprint = perf.group(1).lower()
                order = fingerprint_order.get(fingerprint)
                if not order:
                    unmatched += 1
                    continue

                seconds = float(perf.group(3))
                hashes = int(perf.group(4))
                total_hps = float(perf.group(5))
                cpu_hps = float(perf.group(6))
                gpu_hps = float(perf.group(7))
                by_order[order].add(seconds, hashes, total_hps, cpu_hps, gpu_hps)

    return dict(by_order), perf_records, unmatched


def fast_position(order: str) -> str:
    parts = order.split("/")
    try:
        index = parts.index("CN-Fast")
    except ValueError:
        return "none"
    return ("first", "middle", "last")[index] if index < 3 else f"pos{index + 1}"


def rate(value: float) -> str:
    return f"{value / 1000.0:.3f} kH/s" if value >= 1000.0 else f"{value:.2f} H/s"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="log file path(s) or shell-style glob(s)")
    parser.add_argument("--min-samples", type=int, default=1, help="hide CN orders with fewer samples")
    args = parser.parse_args()

    paths = expand_inputs(args.logs)
    if not paths:
        parser.error("no readable log files matched")

    by_order, perf_records, unmatched = parse_files(paths)
    rows = [
        (order, aggregate)
        for order, aggregate in by_order.items()
        if aggregate.samples >= max(1, args.min_samples)
    ]
    rows.sort(key=lambda item: item[1].hps)

    print(f"files={len(paths)} perf_records={perf_records} matched={perf_records - unmatched} unmatched={unmatched}")
    print()
    print("Ordered CryptoNight triple performance (slowest first)")
    print("samples   active_s    total_avg    gpu_avg      cpu_avg      min          max          fast_pos  CN order")
    print("-------   --------    ---------    -------      -------      ---          ---          --------  --------")
    for order, agg in rows:
        print(
            f"{agg.samples:7d}   {agg.seconds:8.1f}    {rate(agg.hps):>9}    "
            f"{rate(agg.gpu_hps):>9}    {rate(agg.cpu_hps):>9}    "
            f"{rate(agg.min_hps):>9}    {rate(agg.max_hps):>9}    "
            f"{fast_position(order):>8}  {order}"
        )

    position_totals: dict[str, Aggregate] = defaultdict(Aggregate)
    for order, agg in by_order.items():
        position = fast_position(order)
        # Merge already-aggregated values without losing hash/time weighting.
        merged = position_totals[position]
        merged.samples += agg.samples
        merged.seconds += agg.seconds
        merged.hashes += agg.hashes
        merged.cpu_hashes += agg.cpu_hashes
        merged.gpu_hashes += agg.gpu_hashes
        merged.min_hps = min(merged.min_hps, agg.min_hps)
        merged.max_hps = max(merged.max_hps, agg.max_hps)

    print()
    print("CN-Fast position summary")
    for position in ("first", "middle", "last", "none"):
        agg = position_totals.get(position)
        if not agg or agg.samples == 0:
            continue
        print(
            f"{position:>6}: samples={agg.samples:4d} active={agg.seconds:8.1f}s "
            f"avg={rate(agg.hps)} gpu={rate(agg.gpu_hps)} cpu={rate(agg.cpu_hps)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
