#!/usr/bin/env python3
"""Build a deduplicated GhostRider fingerprint performance history from miner logs.

Accepts directories, individual .log/.txt files, and .zip archives.  The parser
uses only Python's standard library so it can run on a normal miner host.

Outputs:
  rotation_perf.csv       One deduplicated [rotation perf] observation per row.
  batch_summary.csv       Per-source/per-fingerprint/per-GPU/per-batch scan stats.
  fingerprint_summary.csv Historical performance summary keyed by fingerprint.
  batch_comparison.csv    Same-hardware/same-fingerprint batch alternatives.
  preferred_batch.csv     Ranked preference with OBSERVED/CANDIDATE/VALIDATED state.
  validated_batch.csv     Only preferences that satisfy validation thresholds.
  coverage_report.csv     Dataset maturity and fingerprint coverage summary.
  family_batch_comparison.csv
                          Cross-fingerprint CN-family batch evidence.
  preferred_family_batch.csv
                          Read-only family-level batch preference candidates.
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

GPU_INFO_RE = re.compile(
    r"GPU\s+(\d+):\s+(.+?)\s+\|\s+"
    r"CC\s+([\d.]+)\s+\|\s+"
    r"SMs\s+(\d+)\s+\|\s+"
    r"warp\s+(\d+)\s+\|\s+"
    r"([\d.]+)\s+GiB"
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
    gpu_name: str
    compute_capability: str
    sms: int | None
    warp_size: int | None
    vram_gib: float | None
    hardware_signature: str
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
            self.hardware_signature,
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


def make_hardware_signature(
    name: str,
    compute_capability: str,
    sms: int | None,
    warp_size: int | None,
    vram_gib: float | None,
) -> str:
    """Return a stable, hardware-neutral device signature from log metadata."""
    parts = [
        name.strip() or "unknown",
        f"cc={compute_capability or 'unknown'}",
        f"sms={sms if sms is not None else 'unknown'}",
        f"warp={warp_size if warp_size is not None else 'unknown'}",
        f"vram={vram_gib:.2f}GiB" if vram_gib is not None else "vram=unknown",
    ]
    return "|".join(parts)


def parse_sources(inputs: list[Path]) -> tuple[list[RotationPerf], list[BatchSample]]:
    rotations: list[RotationPerf] = []
    batches: list[BatchSample] = []

    for source, handle in iter_sources(inputs):
        latency_target_ms: float | None = None
        min_batch: int | None = None
        cn_by_fingerprint: dict[str, str] = {}
        gpu_info: dict[int, tuple[str, str, int, int, float, str]] = {}

        for raw_line in handle:
            line = raw_line.strip()

            target = TARGET_RE.search(line)
            if target:
                latency_target_ms = float(target.group(1))
                min_batch = int(target.group(2))

            ghost = GHOSTRIDER_RE.search(line)
            if ghost:
                cn_by_fingerprint[ghost.group(1).lower()] = ghost.group(2).strip()

            gpu_match = GPU_INFO_RE.search(line)
            if gpu_match:
                gpu_id, name, cc, sms, warp, vram = gpu_match.groups()
                gpu_id_i = int(gpu_id)
                sms_i = int(sms)
                warp_i = int(warp)
                vram_f = float(vram)
                signature = make_hardware_signature(
                    name, cc, sms_i, warp_i, vram_f
                )
                gpu_info[gpu_id_i] = (
                    name.strip(), cc, sms_i, warp_i, vram_f, signature
                )

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
                gpu_i = int(gpu)
                info = gpu_info.get(gpu_i)
                if info is None:
                    gpu_name = "unknown"
                    cc = "unknown"
                    sms_i = None
                    warp_i = None
                    vram_f = None
                    signature = make_hardware_signature(
                        gpu_name, cc, sms_i, warp_i, vram_f
                    )
                else:
                    gpu_name, cc, sms_i, warp_i, vram_f, signature = info

                batches.append(
                    BatchSample(
                        source=source,
                        fingerprint=fingerprint.lower(),
                        gpu=gpu_i,
                        gpu_name=gpu_name,
                        compute_capability=cc,
                        sms=sms_i,
                        warp_size=warp_i,
                        vram_gib=vram_f,
                        hardware_signature=signature,
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
    parser.add_argument(
        "--candidate-scans",
        type=int,
        default=8,
        help="Minimum scans for CANDIDATE state (default: 8).",
    )
    parser.add_argument(
        "--validated-scans",
        type=int,
        default=20,
        help="Minimum scans for VALIDATED state (default: 20).",
    )
    parser.add_argument(
        "--validated-sources",
        type=int,
        default=2,
        help="Minimum independent source logs for VALIDATED state (default: 2).",
    )
    parser.add_argument(
        "--min-win-pct",
        type=float,
        default=3.0,
        help="Minimum effective-H/s win over runner-up for VALIDATED (default: 3.0).",
    )
    parser.add_argument(
        "--max-stale-regression-pct",
        type=float,
        default=1.0,
        help="Maximum stale-rate regression in percentage points for VALIDATED (default: 1.0).",
    )
    parser.add_argument(
        "--family-min-fingerprints",
        type=int,
        default=20,
        help="Minimum distinct exact fingerprints for a family candidate (default: 20).",
    )
    parser.add_argument(
        "--family-min-scans",
        type=int,
        default=100,
        help="Minimum scans for a family candidate (default: 100).",
    )
    args = parser.parse_args()

    if args.candidate_scans < 1:
        parser.error("--candidate-scans must be >= 1")
    if args.validated_scans < args.candidate_scans:
        parser.error("--validated-scans must be >= --candidate-scans")
    if args.validated_sources < 1:
        parser.error("--validated-sources must be >= 1")
    if args.min_win_pct < 0.0:
        parser.error("--min-win-pct must be >= 0")
    if args.max_stale_regression_pct < 0.0:
        parser.error("--max-stale-regression-pct must be >= 0")
    if args.family_min_fingerprints < 2:
        parser.error("--family-min-fingerprints must be >= 2")
    if args.family_min_scans < 1:
        parser.error("--family-min-scans must be >= 1")

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
                b.source, b.fingerprint, b.gpu, b.hardware_signature,
                b.gpu_name, b.compute_capability, b.sms, b.warp_size,
                b.vram_gib, b.cn, b.batch, b.latency_target_ms, b.min_batch,
            )
        ].append(b)

    batch_rows = []
    for key, samples in sorted(grouped_batches.items()):
        (
            source, fingerprint, gpu, hardware_signature, gpu_name, cc,
            sms, warp_size, vram_gib, cn, batch, target, floor
        ) = key
        stale_scans = sum(s.stale for s in samples)
        batch_rows.append(
            (
                source,
                fingerprint,
                gpu,
                hardware_signature,
                gpu_name,
                cc,
                "" if sms is None else sms,
                "" if warp_size is None else warp_size,
                "" if vram_gib is None else f"{vram_gib:.2f}",
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
            "source", "fingerprint", "gpu", "hardware_signature",
            "gpu_name", "compute_capability", "sms", "warp_size", "vram_gib",
            "cn", "batch", "scans",
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
        by_compare[
            (b.hardware_signature, b.fingerprint, b.cn, b.batch)
        ].append(b)

    comparison_rows = []
    comparison_stats: dict[tuple, dict[str, float | int | str]] = {}

    for (hardware_signature, fingerprint, cn, batch), samples in sorted(
        by_compare.items()
    ):
        sources = {s.source for s in samples}
        devices = {(s.source, s.gpu) for s in samples}
        stale_scans = sum(s.stale for s in samples)
        total_wall_ms = sum(s.wall_ms for s in samples)
        nonstale_hashes = sum(s.hashes for s in samples if not s.stale)
        effective_hps = (
            1000.0 * nonstale_hashes / total_wall_ms
            if total_wall_ms > 0.0 else math.nan
        )

        stats = {
            "hardware_signature": hardware_signature,
            "fingerprint": fingerprint,
            "cn": cn,
            "batch": batch,
            "scans": len(samples),
            "sources": len(sources),
            "devices": len(devices),
            "stale_scans": stale_scans,
            "stale_pct": 100.0 * stale_scans / len(samples),
            "avg_scan_ms": mean(s.scan_ms for s in samples),
            "avg_scan_hps": mean(
                1000.0 * s.hashes / s.scan_ms
                for s in samples if s.scan_ms > 0.0
            ),
            "effective_hps": effective_hps,
            "avg_useful_device_pct": mean(
                s.useful_device_pct for s in samples
            ),
        }
        comparison_stats[
            (hardware_signature, fingerprint, cn, batch)
        ] = stats

        comparison_rows.append(
            (
                hardware_signature,
                fingerprint,
                cn,
                batch,
                len(samples),
                len(sources),
                len(devices),
                stale_scans,
                f"{stats['stale_pct']:.4f}",
                f"{stats['avg_scan_ms']:.3f}",
                f"{stats['avg_scan_hps']:.3f}",
                f"{effective_hps:.3f}",
                f"{stats['avg_useful_device_pct']:.4f}",
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
            "hardware_signature", "fingerprint", "cn", "batch", "scans",
            "sources", "devices", "stale_scans", "stale_pct",
            "avg_scan_ms", "avg_scan_hps", "effective_hps",
            "avg_useful_device_pct", "min_batch_values",
        ],
        comparison_rows,
    )

    # Build a conservative evidence ladder.  Nothing here is hardware-specific:
    # every decision is scoped to hardware signature + fingerprint + CN mix.
    by_preference: dict[tuple, list[dict[str, float | int | str]]] = defaultdict(list)
    for stats in comparison_stats.values():
        by_preference[
            (
                str(stats["hardware_signature"]),
                str(stats["fingerprint"]),
                str(stats["cn"]),
            )
        ].append(stats)

    preferred_rows = []
    validated_rows = []
    state_counts = {"OBSERVED": 0, "CANDIDATE": 0, "VALIDATED": 0}
    fingerprints_by_state = {
        "OBSERVED": set(),
        "CANDIDATE": set(),
        "VALIDATED": set(),
    }

    for (hardware_signature, fingerprint, cn), all_candidates in sorted(
        by_preference.items()
    ):
        if not all_candidates:
            continue

        ranked = sorted(
            all_candidates,
            key=lambda x: (
                float(x["effective_hps"]),
                -float(x["stale_pct"]),
                int(x["scans"]),
            ),
            reverse=True,
        )
        best = ranked[0]

        qualified = [
            x for x in ranked
            if int(x["scans"]) >= args.candidate_scans
        ]
        runner_up = qualified[1] if len(qualified) >= 2 else None

        state = "OBSERVED"
        if len(qualified) >= 2 and best in qualified:
            state = "CANDIDATE"

        improvement_pct = math.nan
        runner_up_batch = ""
        runner_up_effective_hps = math.nan
        runner_up_stale_pct = math.nan

        if runner_up is not None:
            runner_up_batch = int(runner_up["batch"])
            runner_up_effective_hps = float(runner_up["effective_hps"])
            runner_up_stale_pct = float(runner_up["stale_pct"])
            if runner_up_effective_hps > 0.0:
                improvement_pct = (
                    (float(best["effective_hps"]) - runner_up_effective_hps)
                    / runner_up_effective_hps
                    * 100.0
                )

            enough_validation_data = (
                int(best["scans"]) >= args.validated_scans
                and int(best["sources"]) >= args.validated_sources
            )
            clear_win = (
                math.isfinite(improvement_pct)
                and improvement_pct >= args.min_win_pct
            )
            stale_is_safe = (
                float(best["stale_pct"])
                <= runner_up_stale_pct + args.max_stale_regression_pct
            )

            if (
                state == "CANDIDATE"
                and enough_validation_data
                and clear_win
                and stale_is_safe
            ):
                state = "VALIDATED"

        confidence = "low"
        scans = int(best["scans"])
        sources = int(best["sources"])
        if state == "VALIDATED":
            confidence = "high"
        elif state == "CANDIDATE":
            confidence = "medium"

        row = (
            hardware_signature,
            fingerprint,
            cn,
            state,
            int(best["batch"]),
            scans,
            sources,
            int(best["devices"]),
            f"{float(best['effective_hps']):.3f}",
            f"{float(best['stale_pct']):.4f}",
            f"{float(best['avg_scan_ms']):.3f}",
            runner_up_batch,
            "" if not math.isfinite(runner_up_effective_hps)
                else f"{runner_up_effective_hps:.3f}",
            "" if not math.isfinite(runner_up_stale_pct)
                else f"{runner_up_stale_pct:.4f}",
            "" if not math.isfinite(improvement_pct)
                else f"{improvement_pct:.3f}",
            confidence,
            len(all_candidates),
            len(qualified),
        )
        preferred_rows.append(row)
        state_counts[state] += 1
        fingerprints_by_state[state].add(fingerprint)

        if state == "VALIDATED":
            validated_rows.append(row)

    preference_headers = [
        "hardware_signature", "fingerprint", "cn", "state",
        "preferred_batch", "scans", "sources", "devices",
        "best_effective_hps", "best_stale_pct", "avg_scan_ms",
        "runner_up_batch", "runner_up_effective_hps",
        "runner_up_stale_pct", "improvement_pct", "confidence",
        "observed_batches", "qualified_batches",
    ]

    write_csv(
        args.output / "preferred_batch.csv",
        preference_headers,
        preferred_rows,
    )
    write_csv(
        args.output / "validated_batch.csv",
        preference_headers,
        validated_rows,
    )

    unique_batch_fingerprints = {b.fingerprint for b in batches}
    multi_batch_fingerprints = {
        fp
        for fp in unique_batch_fingerprints
        if len({b.batch for b in batches if b.fingerprint == fp}) > 1
    }

    # Cross-fingerprint rotation-family analysis.  This intentionally keeps
    # exact fingerprints as the apples-to-apples unit above, while allowing us
    # to ask whether a batch preference generalizes across many different
    # schedules that share the same canonical CN combination on the same
    # hardware.  Nothing from this table changes live scheduling.
    by_family: dict[tuple, list[BatchSample]] = defaultdict(list)
    for b in batches:
        by_family[
            (b.hardware_signature, b.cn, b.batch)
        ].append(b)

    family_rows = []
    family_stats: dict[tuple, dict[str, float | int | str]] = {}

    for (hardware_signature, cn, batch), samples in sorted(by_family.items()):
        fingerprints = {s.fingerprint for s in samples}
        sources = {s.source for s in samples}
        stale_scans = sum(s.stale for s in samples)
        total_wall_ms = sum(s.wall_ms for s in samples)
        nonstale_hashes = sum(s.hashes for s in samples if not s.stale)
        effective_hps = (
            1000.0 * nonstale_hashes / total_wall_ms
            if total_wall_ms > 0.0 else math.nan
        )

        # Estimate cross-fingerprint dispersion from per-fingerprint effective
        # H/s so a family with wildly varying schedules does not look more
        # stable than it really is.
        per_fp: dict[str, list[BatchSample]] = defaultdict(list)
        for s in samples:
            per_fp[s.fingerprint].append(s)

        per_fp_hps = []
        for fp_samples in per_fp.values():
            fp_wall = sum(s.wall_ms for s in fp_samples)
            fp_hashes = sum(s.hashes for s in fp_samples if not s.stale)
            if fp_wall > 0.0:
                per_fp_hps.append(1000.0 * fp_hashes / fp_wall)

        dispersion_pct = math.nan
        if len(per_fp_hps) >= 2:
            mu = statistics.fmean(per_fp_hps)
            if mu > 0.0:
                dispersion_pct = (
                    statistics.pstdev(per_fp_hps) / mu * 100.0
                )

        stats = {
            "hardware_signature": hardware_signature,
            "cn": cn,
            "batch": batch,
            "fingerprints": len(fingerprints),
            "sources": len(sources),
            "scans": len(samples),
            "stale_pct": 100.0 * stale_scans / len(samples),
            "avg_scan_ms": mean(s.scan_ms for s in samples),
            "effective_hps": effective_hps,
            "dispersion_pct": dispersion_pct,
        }
        family_stats[(hardware_signature, cn, batch)] = stats

        family_rows.append(
            (
                hardware_signature,
                cn,
                batch,
                len(fingerprints),
                len(sources),
                len(samples),
                stale_scans,
                f"{stats['stale_pct']:.4f}",
                f"{stats['avg_scan_ms']:.3f}",
                f"{effective_hps:.3f}",
                "" if not math.isfinite(dispersion_pct)
                    else f"{dispersion_pct:.3f}",
            )
        )

    write_csv(
        args.output / "family_batch_comparison.csv",
        [
            "hardware_signature", "cn", "batch", "fingerprints", "sources",
            "scans", "stale_scans", "stale_pct", "avg_scan_ms",
            "effective_hps", "cross_fingerprint_dispersion_pct",
        ],
        family_rows,
    )

    by_family_pref: dict[tuple, list[dict[str, float | int | str]]] = defaultdict(list)
    for stats in family_stats.values():
        if (
            int(stats["fingerprints"]) >= args.family_min_fingerprints
            and int(stats["scans"]) >= args.family_min_scans
        ):
            by_family_pref[
                (
                    str(stats["hardware_signature"]),
                    str(stats["cn"]),
                )
            ].append(stats)

    family_preferred_rows = []
    for (hardware_signature, cn), candidates in sorted(by_family_pref.items()):
        if not candidates:
            continue
        candidates.sort(
            key=lambda x: (
                float(x["effective_hps"]),
                -float(x["stale_pct"]),
                int(x["fingerprints"]),
            ),
            reverse=True,
        )
        best = candidates[0]
        runner = candidates[1] if len(candidates) >= 2 else None

        improvement_pct = math.nan
        runner_batch = ""
        runner_hps = math.nan
        if runner is not None:
            runner_batch = int(runner["batch"])
            runner_hps = float(runner["effective_hps"])
            if runner_hps > 0.0:
                improvement_pct = (
                    (float(best["effective_hps"]) - runner_hps)
                    / runner_hps
                    * 100.0
                )

        family_preferred_rows.append(
            (
                hardware_signature,
                cn,
                int(best["batch"]),
                int(best["fingerprints"]),
                int(best["sources"]),
                int(best["scans"]),
                f"{float(best['effective_hps']):.3f}",
                f"{float(best['stale_pct']):.4f}",
                "" if not math.isfinite(float(best["dispersion_pct"]))
                    else f"{float(best['dispersion_pct']):.3f}",
                runner_batch,
                "" if not math.isfinite(runner_hps)
                    else f"{runner_hps:.3f}",
                "" if not math.isfinite(improvement_pct)
                    else f"{improvement_pct:.3f}",
                len(candidates),
            )
        )

    write_csv(
        args.output / "preferred_family_batch.csv",
        [
            "hardware_signature", "cn", "preferred_batch",
            "fingerprints", "sources", "scans", "effective_hps",
            "stale_pct", "cross_fingerprint_dispersion_pct",
            "runner_up_batch", "runner_up_effective_hps",
            "improvement_pct", "qualified_batches",
        ],
        family_preferred_rows,
    )

    coverage_rows = [
        ("rotation_fingerprints", len(by_fp)),
        ("batch_fingerprints", len(unique_batch_fingerprints)),
        ("multi_batch_fingerprints", len(multi_batch_fingerprints)),
        ("preference_groups", len(preferred_rows)),
        ("observed_groups", state_counts["OBSERVED"]),
        ("candidate_groups", state_counts["CANDIDATE"]),
        ("validated_groups", state_counts["VALIDATED"]),
        ("observed_fingerprints", len(fingerprints_by_state["OBSERVED"])),
        ("candidate_fingerprints", len(fingerprints_by_state["CANDIDATE"])),
        ("validated_fingerprints", len(fingerprints_by_state["VALIDATED"])),
        ("candidate_scans_threshold", args.candidate_scans),
        ("validated_scans_threshold", args.validated_scans),
        ("validated_sources_threshold", args.validated_sources),
        ("minimum_win_pct", f"{args.min_win_pct:.3f}"),
        (
            "maximum_stale_regression_percentage_points",
            f"{args.max_stale_regression_pct:.3f}",
        ),
        ("family_groups", len(family_stats)),
        ("family_preference_groups", len(family_preferred_rows)),
        ("family_min_fingerprints", args.family_min_fingerprints),
        ("family_min_scans", args.family_min_scans),
    ]
    write_csv(
        args.output / "coverage_report.csv",
        ["metric", "value"],
        coverage_rows,
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
    print(
        "Fingerprints with multiple observed batch sizes: "
        f"{len(multi_batch_fingerprints):,}"
    )
    print(
        "Preference groups: "
        f"{len(preferred_rows):,} "
        f"(OBSERVED={state_counts['OBSERVED']:,}, "
        f"CANDIDATE={state_counts['CANDIDATE']:,}, "
        f"VALIDATED={state_counts['VALIDATED']:,})"
    )
    print(
        "Validated unique fingerprints: "
        f"{len(fingerprints_by_state['VALIDATED']):,}"
    )
    print(
        "Cross-fingerprint family preference groups: "
        f"{len(family_preferred_rows):,}"
    )
    print(f"Wrote analysis to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
