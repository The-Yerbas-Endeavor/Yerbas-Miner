#!/usr/bin/env python3
"""Summarize YERBAS_PRODUCTION_LATENCY_TELEMETRY output."""

from __future__ import annotations

import argparse
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
BATCH_RE = re.compile(
    r"\[latency batch\] GPU (?P<gpu>\d+) "
    r"job=(?P<job>\S+) generation=(?P<generation>\d+) "
    r"rotation=(?P<rotation>[0-9a-fA-F]+) CN=(?P<cn>\S+) "
    r"hashes=(?P<hashes>\d+) queue_ms=(?P<queue>[0-9.]+) "
    r"scan_ms=(?P<scan>[0-9.]+) wall_ms=(?P<wall>[0-9.]+) "
    r"stale=(?P<stale>yes|no)"
)
JOB_RE = re.compile(
    r"\[latency job\] new_job=(?P<new>\S+) generation=(?P<generation>\d+) "
    r"prior_job=(?P<prior>\S+) prior_lifetime_ms=(?P<life>[0-9.]+) "
    r"clean=(?P<clean>yes|no)"
)


@dataclass
class Agg:
    batches: int = 0
    stale_batches: int = 0
    hashes: int = 0
    useful_hashes: int = 0
    stale_hashes: int = 0
    queue_ms: float = 0.0
    scan_ms: float = 0.0
    wall_ms: float = 0.0
    stale_scan_ms: float = 0.0

    def add(self, hashes: int, queue_ms: float, scan_ms: float, wall_ms: float, stale: bool) -> None:
        self.batches += 1
        self.hashes += hashes
        self.queue_ms += queue_ms
        self.scan_ms += scan_ms
        self.wall_ms += wall_ms
        if stale:
            self.stale_batches += 1
            self.stale_hashes += hashes
            self.stale_scan_ms += scan_ms
        else:
            self.useful_hashes += hashes


def pct(num: float, den: float) -> float:
    return 100.0 * num / den if den else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", type=Path)
    args = ap.parse_args()

    by_gpu: dict[int, Agg] = defaultdict(Agg)
    by_cn: dict[str, Agg] = defaultdict(Agg)
    by_batch: dict[int, Agg] = defaultdict(Agg)
    by_gpu_batch: dict[tuple[int, int], Agg] = defaultdict(Agg)
    job_lifetimes_ms: list[float] = []
    clean_jobs = 0
    batches = 0

    for path in args.logs:
        for raw in path.read_text(errors="replace").splitlines():
            line = ANSI_RE.sub("", raw)
            m = BATCH_RE.search(line)
            if m:
                gpu = int(m["gpu"])
                hashes = int(m["hashes"])
                queue_ms = float(m["queue"])
                scan_ms = float(m["scan"])
                wall_ms = float(m["wall"])
                stale = m["stale"] == "yes"
                cn = "/".join(sorted(m["cn"].split("/")))
                by_gpu[gpu].add(hashes, queue_ms, scan_ms, wall_ms, stale)
                by_cn[cn].add(hashes, queue_ms, scan_ms, wall_ms, stale)
                by_batch[hashes].add(hashes, queue_ms, scan_ms, wall_ms, stale)
                by_gpu_batch[(gpu, hashes)].add(hashes, queue_ms, scan_ms, wall_ms, stale)
                batches += 1
                continue

            j = JOB_RE.search(line)
            if j:
                life = float(j["life"])
                if j["prior"] != "none" and life > 0.0:
                    job_lifetimes_ms.append(life)
                if j["clean"] == "yes":
                    clean_jobs += 1

    if not batches:
        print("No [latency batch] records found.")
        return 2

    total = Agg()
    for agg in by_gpu.values():
        total.batches += agg.batches
        total.stale_batches += agg.stale_batches
        total.hashes += agg.hashes
        total.useful_hashes += agg.useful_hashes
        total.stale_hashes += agg.stale_hashes
        total.queue_ms += agg.queue_ms
        total.scan_ms += agg.scan_ms
        total.wall_ms += agg.wall_ms
        total.stale_scan_ms += agg.stale_scan_ms

    print("=== Yerbas production-latency summary ===")
    print(f"GPU batches: {total.batches:,} | stale batches: {total.stale_batches:,} "
          f"({pct(total.stale_batches, total.batches):.2f}%)")
    print(f"GPU hashes completed: {total.hashes:,}")
    print(f"Useful hashes: {total.useful_hashes:,} ({pct(total.useful_hashes, total.hashes):.2f}%)")
    print(f"Stale hashes:  {total.stale_hashes:,} ({pct(total.stale_hashes, total.hashes):.2f}%)")
    print(f"GPU scan time: {total.scan_ms / 1000.0:.1f}s | stale scan time: "
          f"{total.stale_scan_ms / 1000.0:.1f}s ({pct(total.stale_scan_ms, total.scan_ms):.2f}%)")
    if total.useful_hashes:
        upper = 100.0 * total.stale_hashes / total.useful_hashes
        print(f"Theoretical upper-bound uplift if every stale hash were recoverable: +{upper:.2f}%")
    print()

    if job_lifetimes_ms:
        vals = sorted(job_lifetimes_ms)
        print("Job lifetime:")
        print(f"  samples={len(vals)} median={statistics.median(vals)/1000.0:.2f}s "
              f"mean={statistics.fmean(vals)/1000.0:.2f}s "
              f"p10={vals[max(0, math.ceil(len(vals)*0.10)-1)]/1000.0:.2f}s "
              f"p90={vals[max(0, math.ceil(len(vals)*0.90)-1)]/1000.0:.2f}s "
              f"clean_notifications={clean_jobs}")
        print()

    print("Per GPU:")
    for gpu, agg in sorted(by_gpu.items()):
        raw_hps = agg.hashes * 1000.0 / agg.scan_ms if agg.scan_ms else 0.0
        useful_hps = agg.useful_hashes * 1000.0 / agg.scan_ms if agg.scan_ms else 0.0
        print(f"  GPU {gpu}: batches={agg.batches} stale={agg.stale_batches} "
              f"hashes={agg.hashes:,} stale_hashes={pct(agg.stale_hashes, agg.hashes):.2f}% "
              f"raw={raw_hps:.2f} H/s useful={useful_hps:.2f} H/s "
              f"avg_scan={agg.scan_ms/max(1,agg.batches):.2f}ms "
              f"avg_queue={agg.queue_ms/max(1,agg.batches):.3f}ms")
    print()

    print("Per CN combination (sorted by stale scan time):")
    for cn, agg in sorted(by_cn.items(), key=lambda kv: kv[1].stale_scan_ms, reverse=True):
        print(f"  {cn:42s} batches={agg.batches:5d} stale={agg.stale_batches:4d} "
              f"stale_hashes={pct(agg.stale_hashes, agg.hashes):6.2f}% "
              f"avg_scan={agg.scan_ms/max(1,agg.batches):8.2f}ms "
              f"stale_scan={agg.stale_scan_ms/1000.0:8.2f}s")
    print()
    print("Per batch size:")
    total_stale_hashes = max(1, total.stale_hashes)
    for hashes, agg in sorted(by_batch.items()):
        raw_hps = agg.hashes * 1000.0 / agg.scan_ms if agg.scan_ms else 0.0
        useful_hps = agg.useful_hashes * 1000.0 / agg.scan_ms if agg.scan_ms else 0.0
        print(f"  batch={hashes:6d} batches={agg.batches:5d} stale={agg.stale_batches:4d} "
              f"stale_hashes={agg.stale_hashes:8d} "
              f"share_of_all_stale={pct(agg.stale_hashes, total_stale_hashes):6.2f}% "
              f"raw={raw_hps:8.2f} H/s useful={useful_hps:8.2f} H/s "
              f"avg_scan={agg.scan_ms/max(1,agg.batches):8.2f}ms "
              f"stale_scan={agg.stale_scan_ms/1000.0:8.2f}s")
    print()
    print("Per GPU / batch size (stale only):")
    for (gpu, hashes), agg in sorted(by_gpu_batch.items()):
        if not agg.stale_batches:
            continue
        print(f"  GPU {gpu} batch={hashes:6d} stale={agg.stale_batches:4d} "
              f"stale_hashes={agg.stale_hashes:8d} "
              f"stale_scan={agg.stale_scan_ms/1000.0:8.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
