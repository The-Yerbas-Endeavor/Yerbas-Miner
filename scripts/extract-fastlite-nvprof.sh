#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DIR="${1:-}"
if [[ -z "$DIR" ]]; then
    echo "Usage: bash scripts/extract-fastlite-nvprof.sh logs/fastlite-profiler-YYYYMMDD-HHMMSS"
    exit 2
fi

if [[ ! -d "$DIR" ]]; then
    echo "ERROR: directory not found: $DIR" >&2
    exit 2
fi

NVPROF_BIN="$(command -v nvprof 2>/dev/null || true)"
if [[ -z "$NVPROF_BIN" ]]; then
    NVPROF_BIN="$(find /usr/local -maxdepth 3 -type f -name nvprof -path '*/cuda*/bin/nvprof' 2>/dev/null | sort -V | tail -1 || true)"
fi
if [[ -z "$NVPROF_BIN" ]]; then
    echo "ERROR: nvprof not found." >&2
    exit 2
fi

shopt -s nullglob
profiles=("$DIR"/*.nvprof)
if [[ "${#profiles[@]}" -eq 0 ]]; then
    echo "ERROR: no .nvprof profiles found in $DIR" >&2
    exit 2
fi

echo "nvprof: $NVPROF_BIN"
echo "Directory: $DIR"
echo

for profile in "${profiles[@]}"; do
    base="${profile%.nvprof}"
    size="$(stat -c %s "$profile" 2>/dev/null || echo 0)"
    echo "$(basename "$profile"): $size bytes"

    if [[ "$size" -le 0 ]]; then
        echo "  EMPTY - no data to decode"
        continue
    fi

    "$NVPROF_BIN" --import-profile "$profile" \
        --log-file "$base-metrics.txt" >/dev/null 2>&1 || true
    "$NVPROF_BIN" --csv --import-profile "$profile" \
        --log-file "$base-metrics.csv" >/dev/null 2>&1 || true

    echo "  -> $(basename "$base-metrics.txt") ($(stat -c %s "$base-metrics.txt" 2>/dev/null || echo 0) bytes)"
    echo "  -> $(basename "$base-metrics.csv") ($(stat -c %s "$base-metrics.csv" 2>/dev/null || echo 0) bytes)"
done

echo
echo "Metric preview:"
grep -hEi 'stall|occup|eligible|issue|ipc|dram|global|l2|throughput|efficiency' "$DIR"/*-metrics.txt 2>/dev/null | head -120 || true
