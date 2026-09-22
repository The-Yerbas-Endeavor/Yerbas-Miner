#!/usr/bin/env python3
"""Analyze continuous per-job stale-batch crossover telemetry."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
CROSS_RE = re.compile(
    r"\[stale crossover\] generation=(?P<generation>\d+) "
    r"capped_gpu=(?P<gpu>\d+) cap=(?P<cap>\d+)"
    r"(?: minimum_tuned=(?P<minimum_tuned>\d+))?"
)
BATCH_RE = re.compile(
    r"\[latency batch\] GPU (?P<gpu>\d+) "
    r"job=(?P<job>\S+) generation=(?P<generation>\d+) "
    r"rotation=(?P<rotation>[0-9a-fA-F]+) CN=(?P<cn>\S+) "
    r"hashes=(?P<hashes>\d+) queue_ms=(?P<queue>[0-9.]+) "
    r"scan_ms=(?P<scan>[0-9.]+) wall_ms=(?P<wall>[0-9.]+) "
    r"stale=(?P<stale>yes|no)"
)

@dataclass
class Agg:
    batches: int = 0
    stale_batches: int = 0
    hashes: int = 0
    stale_hashes: int = 0
    scan_ms: float = 0.0

    def add(self, hashes: int, scan_ms: float, stale: bool) -> None:
        self.batches += 1
        self.hashes += hashes
        self.scan_ms += scan_ms
        if stale:
            self.stale_batches += 1
            self.stale_hashes += hashes

    @property
    def useful_hashes(self) -> int:
        return self.hashes - self.stale_hashes

    @property
    def raw_hps(self) -> float:
        return self.hashes * 1000.0 / self.scan_ms if self.scan_ms else 0.0

    @property
    def useful_hps(self) -> float:
        return self.useful_hashes * 1000.0 / self.scan_ms if self.scan_ms else 0.0

    @property
    def stale_pct(self) -> float:
        return 100.0 * self.stale_hashes / self.hashes if self.hashes else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    args = ap.parse_args()

    roles: dict[int, tuple[int, int, int]] = {}
    batches: list[dict[str, object]] = []

    for raw in args.log.read_text(errors="replace").splitlines():
        line = ANSI_RE.sub("", raw)

        c = CROSS_RE.search(line)
        if c:
            generation = int(c["generation"])
            roles[generation] = (
                int(c["gpu"]),
                int(c["cap"]),
                int(c["minimum_tuned"] or 0),
            )
            continue

        b = BATCH_RE.search(line)
        if b:
            batches.append({
                "generation": int(b["generation"]),
                "gpu": int(b["gpu"]),
                "hashes": int(b["hashes"]),
                "scan_ms": float(b["scan"]),
                "stale": b["stale"] == "yes",
                "cn": "/".join(sorted(b["cn"].split("/"))),
            })

    if not roles:
        print("No [stale crossover] records found.")
        return 2
    if not batches:
        print("No [latency batch] records found.")
        return 2

    by_generation: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in batches:
        by_generation[int(row["generation"])].append(row)

    eligible: set[int] = set()
    for generation, rows in by_generation.items():
        role = roles.get(generation)
        if role is None:
            continue
        capped_gpu, cap, minimum_tuned = role
        threshold = minimum_tuned if minimum_tuned > 0 else (cap + 1)
        if any(int(r["gpu"]) != capped_gpu and int(r["hashes"]) >= threshold for r in rows):
            eligible.add(generation)

    print("=== Yerbas stale-batch live crossover ===")
    print(f"Crossover generations: {len(roles)}")
    print(f"Eligible large-batch generations: {len(eligible)}")
    if not eligible:
        print("No generation exercised an adaptive batch above the cap.")
        print("Keep the run going longer; the result is not yet informative.")
        return 3
    print()

    by_role: dict[str, Agg] = defaultdict(Agg)
    by_gpu_role: dict[tuple[int, str], Agg] = defaultdict(Agg)
    by_role_batch: dict[tuple[str, int], Agg] = defaultdict(Agg)
    by_cn_role: dict[tuple[str, str], Agg] = defaultdict(Agg)

    for generation in sorted(eligible):
        capped_gpu, cap, minimum_tuned = roles[generation]
        for row in by_generation[generation]:
            gpu = int(row["gpu"])
            role = "capped" if gpu == capped_gpu else "adaptive"
            hashes = int(row["hashes"])
            scan_ms = float(row["scan_ms"])
            stale = bool(row["stale"])
            cn = str(row["cn"])
            by_role[role].add(hashes, scan_ms, stale)
            by_gpu_role[(gpu, role)].add(hashes, scan_ms, stale)
            by_role_batch[(role, hashes)].add(hashes, scan_ms, stale)
            by_cn_role[(cn, role)].add(hashes, scan_ms, stale)

    print("Eligible generation pairs:")
    for generation in sorted(eligible):
        capped_gpu, cap, minimum_tuned = roles[generation]
        capped_rows = [r for r in by_generation[generation] if int(r["gpu"]) == capped_gpu]
        adaptive_rows = [r for r in by_generation[generation] if int(r["gpu"]) != capped_gpu]

        def summarize(rows):
            hashes = sum(int(r["hashes"]) for r in rows)
            stale_hashes = sum(int(r["hashes"]) for r in rows if bool(r["stale"]))
            scan_ms = sum(float(r["scan_ms"]) for r in rows)
            raw = hashes * 1000.0 / scan_ms if scan_ms else 0.0
            useful = (hashes - stale_hashes) * 1000.0 / scan_ms if scan_ms else 0.0
            stale_pct = 100.0 * stale_hashes / hashes if hashes else 0.0
            batches = sorted({int(r["hashes"]) for r in rows})
            cn = str(rows[0]["cn"]) if rows else "?"
            return raw, useful, stale_pct, batches, cn

        c_raw, c_useful, c_stale, c_batches, cn = summarize(capped_rows)
        a_raw, a_useful, a_stale, a_batches, _ = summarize(adaptive_rows)
        delta = 100.0 * (c_useful / a_useful - 1.0) if a_useful else float("inf")
        delta_text = "+inf%" if delta == float("inf") else f"{delta:+.2f}%"
        print(
            f"  gen={generation:3d} CN={cn:42s} capped_gpu={capped_gpu} "
            f"min_tuned={minimum_tuned:5d} "
            f"capped_batch={','.join(map(str, c_batches))} useful={c_useful:8.2f} H/s "
            f"stale={c_stale:6.2f}% | adaptive_batch={','.join(map(str, a_batches))} "
            f"useful={a_useful:8.2f} H/s stale={a_stale:6.2f}% | delta={delta_text}"
        )
    print()

    print("Role totals on eligible generations:")
    for role in ("capped", "adaptive"):
        a = by_role[role]
        print(
            f"  {role:8s} batches={a.batches:4d} stale={a.stale_batches:3d} "
            f"hashes={a.hashes:9d} stale={a.stale_pct:6.2f}% "
            f"raw={a.raw_hps:8.2f} H/s useful={a.useful_hps:8.2f} H/s"
        )

    capped = by_role["capped"]
    adaptive = by_role["adaptive"]
    if adaptive.useful_hps:
        delta = 100.0 * (capped.useful_hps / adaptive.useful_hps - 1.0)
        print(f"  useful-H/s capped vs adaptive: {delta:+.2f}%")
    print()

    print("Same GPU when role changes:")
    for gpu in sorted({g for g, _ in by_gpu_role}):
        for role in ("capped", "adaptive"):
            a = by_gpu_role.get((gpu, role), Agg())
            print(
                f"  GPU {gpu} {role:8s} batches={a.batches:4d} "
                f"stale={a.stale_pct:6.2f}% raw={a.raw_hps:8.2f} "
                f"useful={a.useful_hps:8.2f} H/s"
            )
        a = by_gpu_role.get((gpu, "capped"), Agg())
        b = by_gpu_role.get((gpu, "adaptive"), Agg())
        if a.useful_hps and b.useful_hps:
            delta = 100.0 * (a.useful_hps / b.useful_hps - 1.0)
            print(f"    GPU {gpu} capped/adaptive useful delta: {delta:+.2f}%")
    print()

    print("Batch-size breakdown:")
    for (role, hashes), a in sorted(by_role_batch.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        print(
            f"  {role:8s} batch={hashes:6d} batches={a.batches:4d} "
            f"stale={a.stale_pct:6.2f}% raw={a.raw_hps:8.2f} "
            f"useful={a.useful_hps:8.2f} H/s"
        )
    print()

    print("CN combinations with eligible work:")
    combos = sorted({cn for cn, _ in by_cn_role})
    for cn in combos:
        c = by_cn_role.get((cn, "capped"), Agg())
        a = by_cn_role.get((cn, "adaptive"), Agg())
        if not c.batches and not a.batches:
            continue
        print(
            f"  {cn:42s} capped useful={c.useful_hps:8.2f} H/s "
            f"stale={c.stale_pct:6.2f}% | "
            f"adaptive useful={a.useful_hps:8.2f} H/s stale={a.stale_pct:6.2f}%"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
