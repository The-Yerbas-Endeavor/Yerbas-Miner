#!/usr/bin/env python3
"""Export a compact, read-only learned batch profile from analyzer CSV output.

This tool does not change miner scheduling.  It packages historical evidence in
one machine-readable JSON file so future runtime integration can be reviewed
separately from the learning/analysis step.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value: str, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export read-only Yerbas learned batch evidence to JSON."
    )
    parser.add_argument(
        "analysis_dir",
        type=Path,
        help="Directory produced by analyze-fingerprint-history.py",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <analysis_dir>/learned_profile.json)",
    )
    parser.add_argument(
        "--family-min-win-pct",
        type=float,
        default=3.0,
        help="Minimum family-level effective-H/s advantage to export (default: 3.0).",
    )
    parser.add_argument(
        "--family-max-dispersion-pct",
        type=float,
        default=35.0,
        help="Maximum cross-fingerprint dispersion to export (default: 35.0).",
    )
    args = parser.parse_args()

    analysis_dir = args.analysis_dir.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else analysis_dir / "learned_profile.json"
    )

    validated = read_csv(analysis_dir / "validated_batch.csv")
    families = read_csv(analysis_dir / "preferred_family_batch.csv")

    exact_entries: list[dict] = []
    for row in validated:
        if row.get("state") != "VALIDATED":
            continue
        exact_entries.append(
            {
                "hardware_signature": row["hardware_signature"],
                "fingerprint": row["fingerprint"],
                "cn": row["cn"],
                "batch": as_int(row["preferred_batch"]),
                "evidence": {
                    "scans": as_int(row["scans"]),
                    "sources": as_int(row["sources"]),
                    "devices": as_int(row["devices"]),
                    "effective_hps": finite_or_none(
                        as_float(row["best_effective_hps"])
                    ),
                    "stale_pct": finite_or_none(
                        as_float(row["best_stale_pct"])
                    ),
                    "runner_up_batch": (
                        as_int(row["runner_up_batch"])
                        if row.get("runner_up_batch")
                        else None
                    ),
                    "runner_up_effective_hps": finite_or_none(
                        as_float(row.get("runner_up_effective_hps", ""))
                    ),
                    "improvement_pct": finite_or_none(
                        as_float(row.get("improvement_pct", ""))
                    ),
                    "confidence": row.get("confidence", ""),
                },
            }
        )

    family_entries: list[dict] = []
    skipped_family = {
        "no_competitor": 0,
        "win_margin": 0,
        "dispersion": 0,
    }

    for row in families:
        if as_int(row.get("qualified_batches", "0")) < 2:
            skipped_family["no_competitor"] += 1
            continue

        improvement = as_float(row.get("improvement_pct", ""))
        if not math.isfinite(improvement) or improvement < args.family_min_win_pct:
            skipped_family["win_margin"] += 1
            continue

        dispersion = as_float(row.get("cross_fingerprint_dispersion_pct", ""))
        if (
            not math.isfinite(dispersion)
            or dispersion > args.family_max_dispersion_pct
        ):
            skipped_family["dispersion"] += 1
            continue

        family_entries.append(
            {
                "hardware_signature": row["hardware_signature"],
                "cn": row["cn"],
                "batch": as_int(row["preferred_batch"]),
                "evidence": {
                    "fingerprints": as_int(row["fingerprints"]),
                    "sources": as_int(row["sources"]),
                    "scans": as_int(row["scans"]),
                    "effective_hps": finite_or_none(
                        as_float(row["effective_hps"])
                    ),
                    "stale_pct": finite_or_none(
                        as_float(row["stale_pct"])
                    ),
                    "cross_fingerprint_dispersion_pct": dispersion,
                    "runner_up_batch": (
                        as_int(row["runner_up_batch"])
                        if row.get("runner_up_batch")
                        else None
                    ),
                    "runner_up_effective_hps": finite_or_none(
                        as_float(row.get("runner_up_effective_hps", ""))
                    ),
                    "improvement_pct": improvement,
                },
            }
        )

    exact_entries.sort(
        key=lambda x: (
            x["hardware_signature"],
            x["fingerprint"],
            x["cn"],
        )
    )
    family_entries.sort(
        key=lambda x: (
            x["hardware_signature"],
            x["cn"],
        )
    )

    document = {
        "schema": "yerbas-miner-learned-profile-v1",
        "mode": "read-only-evidence",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "runtime_batch_changes_enabled": False,
            "precedence": ["exact_fingerprint", "rotation_family"],
            "family_min_win_pct": args.family_min_win_pct,
            "family_max_dispersion_pct": args.family_max_dispersion_pct,
        },
        "summary": {
            "exact_validated_entries": len(exact_entries),
            "family_exported_entries": len(family_entries),
            "family_skipped": skipped_family,
        },
        "exact_fingerprint": exact_entries,
        "rotation_family": family_entries,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Exact validated entries: {len(exact_entries):,}")
    print(f"Family entries exported: {len(family_entries):,}")
    print(
        "Family entries skipped: "
        f"no_competitor={skipped_family['no_competitor']:,}, "
        f"win_margin={skipped_family['win_margin']:,}, "
        f"dispersion={skipped_family['dispersion']:,}"
    )
    print("Runtime batch changes: disabled")
    print(f"Wrote read-only learned profile: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
