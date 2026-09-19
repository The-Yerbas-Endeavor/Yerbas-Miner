#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NORMAL_CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}"
NORMAL_DIR="$NORMAL_CACHE_ROOT/yerbas-miner"
mkdir -p "$NORMAL_DIR"

SELECTION="$({
python3 - "$ROOT" "$NORMAL_CACHE_ROOT" <<'PY'
import glob
import os
import sys

root, normal = sys.argv[1:]
roots = [
    os.path.join(root, ".bench-cache", "4lane-splitmul"),
    os.path.join(root, ".bench-cache", "workday-4lane-splitmul"),
    normal,
]

best = None
seen = set()
for cache_root in roots:
    cache_root = os.path.abspath(os.path.expanduser(cache_root))
    if cache_root in seen:
        continue
    seen.add(cache_root)
    for path in glob.glob(os.path.join(cache_root, "yerbas-miner",
                                       "cpu-policy-rev*-default.txt")):
        try:
            parts = open(path, encoding="utf-8").read().split()
            if len(parts) < 13 or parts[0] != "YERBAS_CPU_POLICY":
                continue
            revision = int(parts[1])
            workers = int(parts[2])
            lanes = int(parts[3])
            batch = int(parts[4])
            widths = tuple(map(int, parts[5:11]))
            affinity = int(parts[11])
            throughput = float(parts[12])
            if revision <= 0 or workers <= 0 or batch <= 0:
                continue
            if lanes not in (1, 2, 4) or max(widths) != lanes:
                continue
            if any(w not in (1, 2, 4) for w in widths):
                continue
            if affinity not in (0, 1):
                continue
        except Exception:
            continue
        row = (throughput, cache_root, path)
        if best is None or row[0] > best[0]:
            best = row

if best is None:
    raise SystemExit(1)

print(f"{best[1]}\t{best[2]}\t{best[0]:.12f}")
PY
} 2>/dev/null)"

if [[ -z "$SELECTION" ]]; then
    echo "ERROR: no valid default-mode CPU policy cache found."
    exit 1
fi

IFS=$'\t' read -r BEST_ROOT BEST_FILE BEST_HPS <<< "$SELECTION"
TARGET="$NORMAL_DIR/$(basename "$BEST_FILE")"

target_hps="$(
python3 - "$TARGET" <<'PY'
import os
import sys
path = sys.argv[1]
try:
    parts = open(path, encoding="utf-8").read().split()
    print(float(parts[-1]))
except Exception:
    print(0.0)
PY
)"

echo "Best qualified CPU cache:"
echo "  source: $BEST_FILE"
echo "  H/s:    $BEST_HPS"
echo
echo "Normal production cache:"
echo "  target: $TARGET"
echo "  H/s:    $target_hps"

decision="$(
python3 - "$BEST_HPS" "$target_hps" <<'PY'
import sys
best = float(sys.argv[1])
current = float(sys.argv[2])
print("promote" if best > current else "keep")
PY
)"

if [[ "$decision" != "promote" ]]; then
    echo
    echo "Production cache is already equal or better; no change."
    exit 0
fi

stamp="$(date +%Y%m%d-%H%M%S)"
if [[ -f "$TARGET" ]]; then
    cp -p -- "$TARGET" "$TARGET.pre-promotion-$stamp"
    echo
    echo "Backup: $TARGET.pre-promotion-$stamp"
fi

tmp="$TARGET.tmp.$$"
cp -- "$BEST_FILE" "$tmp"
mv -f -- "$tmp" "$TARGET"

echo
echo "PROMOTED: qualified CPU policy is now the normal production cache."
echo "  $BEST_HPS H/s -> $TARGET"
