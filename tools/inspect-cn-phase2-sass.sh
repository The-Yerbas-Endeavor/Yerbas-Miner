#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BIN="${1:-$ROOT/build-production-latency/yerbas-miner}"
OUT_DIR="${2:-$ROOT/logs/cn-sass-$(date +%Y%m%d-%H%M%S)}"

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: binary not found: $BIN"
    exit 1
fi
if ! command -v cuobjdump >/dev/null 2>&1; then
    echo "ERROR: cuobjdump is required and was not found in PATH."
    exit 1
fi

mkdir -p "$OUT_DIR"
SASS="$OUT_DIR/yerbas-miner.sass.txt"
RES="$OUT_DIR/yerbas-miner.resources.txt"
SUMMARY="$OUT_DIR/cn-phase2-summary.txt"

echo "Dumping CUDA resource usage..."
cuobjdump --dump-resource-usage "$BIN" > "$RES"

echo "Dumping SASS..."
cuobjdump --dump-sass "$BIN" > "$SASS"

python3 - "$SASS" "$RES" "$SUMMARY" <<'PY'
from pathlib import Path
import re
import sys

sass_path, res_path, out_path = map(Path, sys.argv[1:])
sass = sass_path.read_text(errors="replace").splitlines()
resources = res_path.read_text(errors="replace").splitlines()

fn_re = re.compile(r"^\s*Function\s*:\s*(.+?)\s*$")
op_re = re.compile(r"/\*[0-9a-fA-F]+\*/\s+([A-Z][A-Z0-9_.]*)")

functions = {}
current = None
for line in sass:
    m = fn_re.match(line)
    if m:
        current = m.group(1)
        functions.setdefault(current, [])
        continue
    if current is not None:
        functions[current].append(line)

needles = (
    "cryptonight_loop_stage_ttable4",
    "cryptonight_loop_stage_ttable4_cg",
    "cryptonight_loop_stage_ttable",
)

selected = [(name, lines) for name, lines in functions.items()
            if any(n in name for n in needles)]

interesting_prefixes = (
    "XMAD", "IMAD", "IADD", "IADD3", "ISCADD", "LEA",
    "SHF", "SHFL", "LDG", "STG", "LDS", "STS",
    "BAR", "BRA", "SYNC", "LOP", "PRMT"
)

out = []
out.append("Yerbas CN phase-2 SASS inspection")
out.append("=" * 72)
out.append(f"SASS file: {sass_path}")
out.append(f"Resource file: {res_path}")
out.append(f"Matched phase-2 functions: {len(selected)}")
out.append("")

for name, lines in selected:
    counts = {}
    total = 0
    for line in lines:
        m = op_re.search(line)
        if not m:
            continue
        op = m.group(1)
        total += 1
        root = op.split(".")[0]
        if root.startswith(interesting_prefixes):
            counts[root] = counts.get(root, 0) + 1
    out.append(name)
    out.append(f"  instructions={total}")
    for op, count in sorted(counts.items()):
        out.append(f"  {op:10s} {count}")
    out.append("")

out.append("Resource-usage excerpts")
out.append("-" * 72)
for i, line in enumerate(resources):
    low = line.lower()
    if "cryptonight" in low or "ttable4" in low:
        start = max(0, i - 1)
        end = min(len(resources), i + 4)
        out.extend("  " + x for x in resources[start:end])

if not selected:
    out.append("")
    out.append("WARNING: no phase-2 function names matched.")
    out.append("Inspect the full SASS and adjust the function-name needles if nvcc")
    out.append("emitted a different mangled spelling.")

out_path.write_text("\n".join(out) + "\n")
print("\n".join(out))
PY

echo
echo "Full SASS:      $SASS"
echo "Resource usage: $RES"
echo "Summary:        $SUMMARY"
