#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="$ROOT/build-cpu-combo"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

# This diagnostic must consume an already-proven FULL startup CPU policy. Do not
# let stale developer controls trigger fresh tuning or live rotation learning.
unset YERBAS_CPU_RETUNE || true
unset YERBAS_CPU_RUNTIME_LEARN || true
unset YERBAS_CUDA_RETUNE || true
unset YERBAS_CUDA_OVERLAP || true
unset YERBAS_GPU_AUTOTUNE || true
unset YERBAS_GPU_VARIANT_AUTOTUNE || true

# Prefer the completed Sept-16 workday/full-tune cache, then its cloned workday
# cache, then the user's normal cache. An explicit override remains available.
if [[ -n "${YERBAS_CPU_COMBO_XDG_CACHE_HOME:-}" ]]; then
    CACHE_ROOT="$YERBAS_CPU_COMBO_XDG_CACHE_HOME"
else
    CACHE_ROOT=""
    for candidate in \
        "$ROOT/.bench-cache/4lane-splitmul" \
        "$ROOT/.bench-cache/workday-4lane-splitmul" \
        "${XDG_CACHE_HOME:-$HOME/.cache}"
    do
        if compgen -G "$candidate/yerbas-miner/cpu-policy-rev*-full.txt" >/dev/null; then
            CACHE_ROOT="$candidate"
            break
        fi
    done
fi

if [[ -z "$CACHE_ROOT" ]] || ! compgen -G "$CACHE_ROOT/yerbas-miner/cpu-policy-rev*-full.txt" >/dev/null; then
    echo "ERROR: no cached FULL CPU policy was found."
    echo "Checked likely cache roots:"
    echo "  $ROOT/.bench-cache/4lane-splitmul"
    echo "  $ROOT/.bench-cache/workday-4lane-splitmul"
    echo "  ${XDG_CACHE_HOME:-$HOME/.cache}"
    echo
    echo "Existing CPU policy files:"
    find "$ROOT/.bench-cache" "${XDG_CACHE_HOME:-$HOME/.cache}/yerbas-miner" \
      -type f -name 'cpu-policy-rev*' -print 2>/dev/null | sort || true
    echo
    echo "Refusing to trigger another CPU retune."
    exit 1
fi

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
  " Binary:     $BIN" \
  " Cache root: $CACHE_ROOT" \
  " Log:        $LOG" \
  "" \
  " Requires the existing cached FULL CPU policy as baseline." \
  " Tests all 20 unordered three-CryptoNight combinations." \
  " Does not initialize CUDA, connect to the pool, or write a" \
  " production CPU combination policy." \
  "============================================================" \
  ""

export YERBAS_CPU_COMBO_BENCH=1
export YERBAS_CPU_COMBO_PASSES="${YERBAS_CPU_COMBO_PASSES:-5}"

set +e
script -q -f -e -c "$BIN --tune full" "$LOG"
STATUS=$?
set -e

echo
echo "================ RAW BENCHMARK SUMMARY ================"
grep -E '^\[CPU tune\]|^\[CPU combo result\]|^Combinations tested|^Promoted candidates|^Equal-combo|^Projected CPU gain|^ERROR:|^Fatal:' "$LOG" || true

echo
echo "================ SAFE CONFIRMATION SUMMARY ================"
python3 - "$LOG" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(errors="replace")
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
    promoted = parity == "PASS" and gain >= 2.0 and selected > base
    safe = selected if promoted else base
    rows.append((name, base, selected, gain, widths, promoted, safe))

if len(rows) != 20:
    print(f"ERROR: expected 20 combo results, found {len(rows)}")
    raise SystemExit(2)

base_avg = sum(row[1] for row in rows) / len(rows)
safe_avg = sum(row[6] for row in rows) / len(rows)
safe_gain = (safe_avg / base_avg - 1.0) * 100.0 if base_avg else 0.0

for name, base, selected, gain, widths, promoted, safe in rows:
    verdict = "KEEP" if promoted else "BASELINE"
    print(f"{verdict:8s} | {name:42s} | {gain:+7.2f}% | {widths}")

print()
print(f"Confirmation-safe promotions : {sum(row[5] for row in rows)}/20")
print(f"Equal-combo baseline         : {base_avg:.2f} H/s")
print(f"Equal-combo safe selected    : {safe_avg:.2f} H/s")
print(f"Confirmation-safe CPU gain   : {safe_gain:+.2f}%")
PY

echo "Log: $LOG"
exit "$STATUS"
