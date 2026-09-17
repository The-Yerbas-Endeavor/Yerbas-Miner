#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="$ROOT/build-cpu-combo"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

# This diagnostic must consume an already-proven startup CPU policy. Do not let
# stale developer controls trigger fresh tuning or live rotation learning.
unset YERBAS_CPU_RETUNE || true
unset YERBAS_CPU_RUNTIME_LEARN || true
unset YERBAS_CUDA_RETUNE || true
unset YERBAS_CUDA_OVERLAP || true
unset YERBAS_GPU_AUTOTUNE || true
unset YERBAS_GPU_VARIANT_AUTOTUNE || true

TUNE_MODE="${YERBAS_CPU_COMBO_TUNE_MODE:-default}"
SAFE_GAIN="${YERBAS_CPU_COMBO_SAFE_GAIN:-3.0}"
NORMAL_CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}"
OVERRIDE_CACHE_ROOT="${YERBAS_CPU_COMBO_XDG_CACHE_HOME:-}"

# Find the strongest compatible cached policy instead of accepting whichever
# XDG cache happens to be active. The Sept-16 workday cache is expected to beat
# the older ~/.cache policy on this machine, but this remains hardware-generic.
set +e
CACHE_SELECTION="$({
python3 - "$ROOT" "$NORMAL_CACHE_ROOT" "$OVERRIDE_CACHE_ROOT" "$TUNE_MODE" <<'PY'
import glob
import os
import sys

root, normal, override, mode = sys.argv[1:]
if override:
    roots = [override]
else:
    roots = [
        os.path.join(root, ".bench-cache", "4lane-splitmul"),
        os.path.join(root, ".bench-cache", "workday-4lane-splitmul"),
        normal,
    ]

seen = set()
best = None
for cache_root in roots:
    cache_root = os.path.abspath(os.path.expanduser(cache_root))
    if cache_root in seen:
        continue
    seen.add(cache_root)
    pattern = os.path.join(
        cache_root, "yerbas-miner", f"cpu-policy-rev*-{mode}.txt"
    )
    for path in glob.glob(pattern):
        try:
            parts = open(path, encoding="utf-8").read().split()
            if len(parts) < 13 or parts[0] != "YERBAS_CPU_POLICY":
                continue
            throughput = float(parts[-1])
        except Exception:
            continue
        row = (throughput, cache_root, path)
        if best is None or row[0] > best[0]:
            best = row

if best is None:
    raise SystemExit(1)

throughput, cache_root, path = best
print(f"{cache_root}\t{path}\t{throughput:.6f}")
PY
} 2>/dev/null)"
SELECT_STATUS=$?
set -e

if [[ $SELECT_STATUS -ne 0 || -z "$CACHE_SELECTION" ]]; then
    echo "ERROR: no cached CPU policy compatible with --tune $TUNE_MODE was found."
    echo "Checked likely cache roots:"
    echo "  $ROOT/.bench-cache/4lane-splitmul"
    echo "  $ROOT/.bench-cache/workday-4lane-splitmul"
    echo "  $NORMAL_CACHE_ROOT"
    echo
    echo "Existing CPU policy files:"
    find "$ROOT/.bench-cache" "$NORMAL_CACHE_ROOT/yerbas-miner" \
      -type f -name 'cpu-policy-rev*' -print 2>/dev/null | sort || true
    echo
    echo "Refusing to trigger another CPU retune."
    exit 1
fi

IFS=$'\t' read -r CACHE_ROOT CACHE_FILE CACHE_HPS <<< "$CACHE_SELECTION"
export XDG_CACHE_HOME="$CACHE_ROOT"

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    cmake -S . -B "$BUILD_DIR" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DYERBAS_ENABLE_CUDA=OFF \
        -DYERBAS_NATIVE_CPU=OFF
fi

cmake --build "$BUILD_DIR" --target yerbas-miner -j"$(nproc)"

BIN="$BUILD_DIR/yerbas-miner"
LOG="$LOG_DIR/cpu-cn-combo-$(date +%Y%m%d-%H%M%S).log"

printf '%s\n' \
  "============================================================" \
  " YERBAS CPU CN-COMBINATION BENCHMARK" \
  "============================================================" \
  " Binary:       $BIN" \
  " Tune mode:    $TUNE_MODE" \
  " Cache root:   $CACHE_ROOT" \
  " Cache policy: $CACHE_FILE" \
  " Cached H/s:   $CACHE_HPS" \
  " Safe gate:    >=${SAFE_GAIN}% confirmed" \
  " Log:          $LOG" \
  "" \
  " Uses the strongest compatible cached CPU policy as baseline." \
  " Tests all 20 unordered three-CryptoNight combinations." \
  " Does not initialize CUDA, connect to the pool, or write a" \
  " production CPU combination policy." \
  "============================================================" \
  ""

export YERBAS_CPU_COMBO_BENCH=1
export YERBAS_CPU_COMBO_PASSES="${YERBAS_CPU_COMBO_PASSES:-7}"

set +e
script -q -f -e -c "$BIN --tune $TUNE_MODE" "$LOG"
STATUS=$?
set -e

echo
echo "================ RAW BENCHMARK SUMMARY ================"
grep -E '^\[CPU tune\]|^\[CPU combo result\]|^Combinations tested|^Promoted candidates|^Equal-combo|^Projected CPU gain|^ERROR:|^Fatal:' "$LOG" || true

echo
echo "================ HARDENED CONFIRMATION SUMMARY ================"
python3 - "$LOG" "$SAFE_GAIN" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(errors="replace")
safe_gate = float(sys.argv[2])

baseline_widths_match = re.search(r"^Baseline widths\s*: ([0-9/]+)$", text, re.MULTILINE)
baseline_widths = baseline_widths_match.group(1) if baseline_widths_match else "?"

pattern = re.compile(
    r"^\[CPU combo result\] (.*?) \| base=([0-9.]+) H/s \| "
    r"selected=([0-9.]+) H/s \| gain=([+-]?[0-9.]+)% \| "
    r"widths=([0-9/]+) \| parity=(PASS|FAIL)$",
    re.MULTILINE,
)

rows = []
for match in pattern.finditer(text):
    name, base, selected, gain, widths, parity = match.groups()
    base = float(base)
    selected = float(selected)
    gain = float(gain)
    promoted = (
        parity == "PASS"
        and gain >= safe_gate
        and selected > base
    )
    safe = selected if promoted else base
    effective_widths = widths if promoted else baseline_widths
    rows.append((name, base, selected, gain, widths, parity, promoted, safe, effective_widths))

if len(rows) != 20:
    print(f"ERROR: expected 20 combo results, found {len(rows)}")
    raise SystemExit(2)

base_avg = sum(row[1] for row in rows) / len(rows)
safe_avg = sum(row[7] for row in rows) / len(rows)
safe_gain = (safe_avg / base_avg - 1.0) * 100.0 if base_avg else 0.0

for name, base, selected, gain, widths, parity, promoted, safe, effective_widths in rows:
    verdict = "KEEP" if promoted else "BASELINE"
    print(
        f"{verdict:8s} | {name:42s} | {gain:+7.2f}% | "
        f"effective={effective_widths} | measured={widths}"
    )

print()
print(f"Safe confirmed gate          : >= {safe_gate:.2f}%")
print(f"Hardened promotions         : {sum(row[6] for row in rows)}/20")
print(f"Equal-combo baseline        : {base_avg:.2f} H/s")
print(f"Equal-combo safe selected   : {safe_avg:.2f} H/s")
print(f"Hardened projected CPU gain : {safe_gain:+.2f}%")
PY

echo "Log: $LOG"
exit "$STATUS"
