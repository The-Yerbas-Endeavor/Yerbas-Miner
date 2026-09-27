#!/usr/bin/env python3
"""Build a deduplicated GhostRider fingerprint performance history from miner logs.

Accepts directories, individual .log/.txt files, and .zip archives.  The parser
uses only Python's standard library so it can run on a normal miner host.

Outputs:
  rotation_perf.csv       One deduplicated [rotation perf] observation per row.
  batch_summary.csv       Per-source/per-fingerprint/per-GPU/per-batch scan stats.
  fingerprint_summary.csv Historical performance summary keyed by fingerprint.
  batch_comparison.csv    Same-fingerprint batch alternatives suitable for A/B/C.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import re
import statistics
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, TextIO


ROTATION_RE = re.compile(
    r"\[rotation perf\]\s+"
    r"fingerprint=([0-9a-fA-F]+)\s+\|\s+"
    r"sample=(\d+)\s+\|\s+"
    r"duration=([\d.]+)s\s+\|\s+"
    r"hashes=(\d+)\s+\|\s+"
    r"total=([\d.]+) H/s\s+\|\s+"
    r"cpu=([\d.]+) H/s\s+\|\s+"
    r"gpu=([\d.]+) H/s\s+\|\s+"
    r"fingerprint_avg=([\d.]+) H/s\s+\|\s+"
    r"ewma=([\d.]+) H/s"
)

BATCH_RE = re.compile(
    r"\[latency batch\]\s+GPU\s+(\d+)\s+.*?"
    r"rotation=([0-9a-fA-F]+)\s+"
    r"CN=([^\s]+)\s+"
    r"hashes=(\d+)\s+"
    r"queue_ms=([\d.]+)\s+"
    r"scan_ms=([\d.]+)\s+"
    r"wall_ms=([\d.]+)\s+"
    r"stale=(yes|no)\s+"
    r"useful_device_pct=([\d.]+)"
)

TARGET_RE = re.compile(
    r"\[latency-target\].*?"
    r"target_ms=([\d.]+).*?"
    r"min_batch=(\d+)"
)

GHOSTRIDER_RE = re.compile(
    r"\[GhostRider\]\s+rotation=([0-9a-fA-F]+)\s+\|\s+CN=(.+)$"
)


@dataclass(frozen=True)
class RotationPerf:
    source: str
    fingerprint: str
    sample: int
    duration_s: float
    hashes: int
    total_hps: float
    cpu_hps: float
    gpu_hps: float
    fingerprint_avg_hps: float
    ewma_hps: float
    latency_target_ms: float | None
    min_batch: int | None

    def dedup_key(self) -> tuple:
        # Companion logs often contain the exact same rotation record.
        return (
            self.fingerprint,
            self.sample,
            round(self.duration_s, 3),
            self.hashes,
            round(self.total_hps, 3),
            round(self.cpu_hps, 3),
            round(self.gpu_hps, 3),
            round(self.fingerprint_avg_hps, 3),
            round(self.ewma_hps, 3),
        )


@dataclass(frozen=True)
class BatchSample:
    source: str
    fingerprint: str
    gpu: int
    cn: str
    batch: int
    hashes: int
    queue_ms: float
    scan_ms: float
    wall_ms: float
    stale: bool
    useful_device_pct: float
    latency_target_ms: float | None
    min_batch: int | None

    def dedup_key(self) -> tuple:
        # Preserve genuinely repeated scans while eliminating mirrored copies.
        # Floating telemetry makes accidental exact collisions vanishingly rare.
        return (
            self.fingerprint,
            self.gpu,
            self.cn,
            self.batch,
            self.hashes,
            round(self.queue_ms, 3),
            round(self.scan_ms, 3),
            round(self.wall_ms, 3),
            self.stale,
            round(self.useful_device_pct, 3),
        )


def iter_plain_file(path: Path) -> Iterator[tuple[str, TextIO]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        yield str(path), handle


def iter_zip(path: Path) -> Iterator[tuple[str, TextIO]]:
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            suffix = Path(info.filename).suffix.lower()
            if suffix not in {".log", ".txt"}:
                continue
            raw = archive.open(info, "r")
            text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
            try:
                yield f"{path}!{info.filename}", text
            finally:
                text.close()


def iter_sources(inputs: Iterable[Path]) -> Iterator[tuple[str, TextIO]]:
    seen_paths: set[Path] = set()

    for item in inputs:
        item = item.expanduser().resolve()
        if not item.exists():
            print(f"warning: input not found: {item}", file=sys.stderr)
            continue

        if item.is_dir():
            candidates = sorted(
                p for p in item.rglob("*")
                if p.is_file() and p.suffix.lower() in {".log", ".txt", ".zip"}
            )
        else:
            candidates = [item]

        for path in candidates:
            if path in seen_paths:
                continue
            seen_paths.add(path)

            if path.suffix.lower() == ".zip":
                yield from iter_zip(path)
            elif path.suffix.lower() in {".log", ".txt"}:
                yield from iter_plain_file(path)


def parse_sources(inputs: list[Path]) -> tuple[list[RotationPerf], list[BatchSample]]:
    rotations: list[RotationPerf] = []
    batches: list[BatchSample] = []

    for source, handle in iter_sources(inputs):
        latency_target_ms: float | None = None
        min_batch: int | None = None
        cn_by_fingerprint: dict[str, str] = {}

        for raw_line in handle:
            line = raw_line.strip()

            target = TARGET_RE.search(line)
            if target:
                latency_target_ms = float(target.group(1))
                min_batch = int(target.group(2))

            ghost = GHOSTRIDER_RE.search(line)
            if ghost:
                cn_by_fingerprint[ghost.group(1).lower()] = ghost.group(2).strip()

            rotation = ROTATION_RE.search(line)
            if rotation:
                (
                    fingerprint,
                    sample,
                    duration_s,
                    hashes,
                    total_hps,
                    cpu_hps,
                    gpu_hps,
                    fingerprint_avg_hps,
                    ewma_hps,
                ) = rotation.groups()
                rotations.append(
                    RotationPerf(
                        source=source,
                        fingerprint=fingerprint.lower(),
                        sample=int(sample),
                        duration_s=float(duration_s),
                        hashes=int(hashes),
                        total_hps=float(total_hps),
                        cpu_hps=float(cpu_hps),
                        gpu_hps=float(gpu_hps),
                        fingerprint_avg_hps=float(fingerprint_avg_hps),
                        ewma_hps=float(ewma_hps),
                        latency_target_ms=latency_target_ms,
                        min_batch=min_batch,
                    )
                )

            batch = BATCH_RE.search(line)
            if batch:
                (
                    gpu,
                    fingerprint,
                    cn,
                    hashes,
                    queue_ms,
                    scan_ms,
                    wall_ms,
                    stale,
                    useful_device_pct,
                ) = batch.groups()
                batches.append(
                    BatchSample(
                        source=source,
                        fingerprint=fingerprint.lower(),
                        gpu=int(gpu),
                        cn=cn,
                        batch=int(hashes),
                        hashes=int(hashes),
                        queue_ms=float(queue_ms),
                        scan_ms=float(scan_ms),
                        wall_ms=float(wall_ms),
                        stale=stale == "yes",
                        useful_device_pct=float(useful_device_pct),
                        latency_target_ms=latency_target_ms,
                        min_batch=min_batch,
                    )
                )

    return rotations, batches


def dedup(items, key_fn):
    unique = {}
    for item in items:
        unique.setdefault(key_fn(item), item)
    return list(unique.values())


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return statistics.fmean(vals) if vals else math.nan


def write_csv(path: Path, headers: list[str], rows: Iterable[Iterable]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a deduplicated Yerbas fingerprint performance history."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Log files, directories, or .zip archives.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("fingerprint-history"),
        help="Output directory (default: fingerprint-history).",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    rotations_raw, batches_raw = parse_sources(args.inputs)
    rotations = dedup(rotations_raw, RotationPerf.dedup_key)
    batches = dedup(batches_raw, BatchSample.dedup_key)

    rotations.sort(key=lambda r: (r.fingerprint, r.source))
    batches.sort(key=lambda b: (b.fingerprint, b.gpu, b.batch, b.source))

    write_csv(
        args.output / "rotation_perf.csv",
        [
            "source", "fingerprint", "sample", "duration_s", "hashes",
            "total_hps", "cpu_hps", "gpu_hps", "fingerprint_avg_hps",
            "ewma_hps", "latency_target_ms", "min_batch",
        ],
        (
            (
                r.source, r.fingerprint, r.sample, f"{r.duration_s:.3f}",
                r.hashes, f"{r.total_hps:.3f}", f"{r.cpu_hps:.3f}",
                f"{r.gpu_hps:.3f}", f"{r.fingerprint_avg_hps:.3f}",
                f"{r.ewma_hps:.3f}",
                "" if r.latency_target_ms is None else f"{r.latency_target_ms:.1f}",
                "" if r.min_batch is None else r.min_batch,
            )
            for r in rotations
        ),
    )

    grouped_batches: dict[tuple, list[BatchSample]] = defaultdict(list)
    for b in batches:
        grouped_batches[
            (
                b.source, b.fingerprint, b.gpu, b.cn, b.batch,
                b.latency_target_ms, b.min_batch,
            )
        ].append(b)

    batch_rows = []
    for key, samples in sorted(grouped_batches.items()):
        source, fingerprint, gpu, cn, batch, target, floor = key
        stale_scans = sum(s.stale for s in samples)
        batch_rows.append(
            (
                source,
                fingerprint,
                gpu,
                cn,
                batch,
                len(samples),
                stale_scans,
                f"{100.0 * stale_scans / len(samples):.4f}",
                f"{mean(s.scan_ms for s in samples):.3f}",
                f"{mean(s.wall_ms for s in samples):.3f}",
                f"{mean((1000.0 * s.hashes / s.scan_ms) for s in samples if s.scan_ms > 0.0):.3f}",
                f"{mean((1000.0 * s.hashes / s.wall_ms) for s in samples if s.wall_ms > 0.0 and not s.stale):.3f}",
                f"{mean(s.useful_device_pct for s in samples):.4f}",
                "" if target is None else f"{target:.1f}",
                "" if floor is None else floor,
            )
        )

    write_csv(
        args.output / "batch_summary.csv",
        [
            "source", "fingerprint", "gpu", "cn", "batch", "scans",
            "stale_scans", "stale_pct", "avg_scan_ms", "avg_wall_ms",
            "avg_scan_hps", "avg_nonstale_wall_hps",
            "avg_useful_device_pct", "latency_target_ms", "min_batch",
        ],
        batch_rows,
    )

    by_fp: dict[str, list[RotationPerf]] = defaultdict(list)
    for r in rotations:
        by_fp[r.fingerprint].append(r)

    fingerprint_rows = []
    for fingerprint, records in sorted(by_fp.items()):
        fingerprint_rows.append(
            (
                fingerprint,
                len(records),
                len({r.source for r in records}),
                sum(r.duration_s for r in records),
                sum(r.hashes for r in records),
                f"{mean(r.total_hps for r in records):.3f}",
                f"{mean(r.cpu_hps for r in records):.3f}",
                f"{mean(r.gpu_hps for r in records):.3f}",
                f"{min(r.total_hps for r in records):.3f}",
                f"{max(r.total_hps for r in records):.3f}",
                ";".join(
                    str(x)
                    for x in sorted(
                        {r.min_batch for r in records if r.min_batch is not None}
                    )
                ),
            )
        )

    write_csv(
        args.output / "fingerprint_summary.csv",
        [
            "fingerprint", "observations", "sources", "duration_s", "hashes",
            "avg_total_hps", "avg_cpu_hps", "avg_gpu_hps", "min_total_hps",
            "max_total_hps", "min_batch_values",
        ],
        fingerprint_rows,
    )

    by_compare: dict[tuple, list[BatchSample]] = defaultdict(list)
    for b in batches:
        by_compare[(b.fingerprint, b.gpu, b.cn, b.batch)].append(b)

    comparison_rows = []
    for (fingerprint, gpu, cn, batch), samples in sorted(by_compare.items()):
        sources = {s.source for s in samples}
        stale_scans = sum(s.stale for s in samples)
        comparison_rows.append(
            (
                fingerprint,
                gpu,
                cn,
                batch,
                len(samples),
                len(sources),
                stale_scans,
                f"{100.0 * stale_scans / len(samples):.4f}",
                f"{mean(s.scan_ms for s in samples):.3f}",
                f"{mean((1000.0 * s.hashes / s.scan_ms) for s in samples if s.scan_ms > 0.0):.3f}",
                f"{mean((1000.0 * s.hashes / s.wall_ms) for s in samples if s.wall_ms > 0.0 and not s.stale):.3f}",
                f"{mean(s.useful_device_pct for s in samples):.4f}",
                ";".join(
                    str(x)
                    for x in sorted(
                        {s.min_batch for s in samples if s.min_batch is not None}
                    )
                ),
            )
        )

    write_csv(
        args.output / "batch_comparison.csv",
        [
            "fingerprint", "gpu", "cn", "batch", "scans", "sources",
            "stale_scans", "stale_pct", "avg_scan_ms",
            "avg_scan_hps", "avg_nonstale_wall_hps",
            "avg_useful_device_pct", "min_batch_values",
        ],
        comparison_rows,
    )

    print(
        f"Parsed {len(rotations_raw):,} rotation records and "
        f"{len(batches_raw):,} latency batches."
    )
    print(
        f"After exact companion-log dedup: {len(rotations):,} rotations, "
        f"{len(batches):,} latency batches."
    )
    print(f"Unique fingerprints: {len(by_fp):,}")
    multi_batch = {
        fp
        for fp in {b.fingerprint for b in batches}
        if len({b.batch for b in batches if b.fingerprint == fp}) > 1
    }
    print(f"Fingerprints with multiple observed batch sizes: {len(multi_batch):,}")
    print(f"Wrote analysis to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
