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
SASS="$OUT_DIR/cn-phase2.sass.txt"
RES="$OUT_DIR/yerbas-miner.resources.txt"
SUMMARY="$OUT_DIR/cn-phase2-summary.txt"

echo "Dumping CUDA resource usage..."
cuobjdump --dump-resource-usage "$BIN" > "$RES"

echo "Streaming only CryptoNight phase-2 SASS..."
# The release binary contains several real GPU architectures and many templated
# CUDA kernels. A raw cuobjdump --dump-sass can therefore be multiple gigabytes.
# Keep only the three loop-kernel families we actually inspect.
cuobjdump --dump-sass "$BIN" | python3 - "$SASS" <<'PYFILTER'
from pathlib import Path
import re
import sys

out_path = Path(sys.argv[1])
needles = (
    "cryptonight_loop_stage_ttable4_coalesced",
    "cryptonight_loop_stage_ttable4_cg",
    "cryptonight_loop_stage_ttable2_tile64",
)

fn_re = re.compile(r"^\\s*Function\\s*:\\s*(.+?)\\s*$")
selected = False
written = 0

with out_path.open("w", encoding="utf-8") as out:
    for line in sys.stdin:
        m = fn_re.match(line)
        if m:
            selected = any(n in m.group(1) for n in needles)
        if selected:
            out.write(line)
            written += 1

print(f"Kept {written} phase-2 SASS lines in {out_path}", file=sys.stderr)
PYFILTER

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

# cuobjdump resource output varies a little by CUDA toolkit. Preserve the raw
# matching excerpts, then extract the common REG/SHARED/LOCAL/STACK fields when
# available so the summary immediately exposes occupancy/spill pressure.
resource_chunks = []
for i, line in enumerate(resources):
    low = line.lower()
    if "cryptonight" in low or "ttable4" in low:
        start = max(0, i - 1)
        end = min(len(resources), i + 5)
        chunk = resources[start:end]
        resource_chunks.append(chunk)
        out.extend("  " + x for x in chunk)

out.append("")
out.append("Resource-pressure hints")
out.append("-" * 72)

field_patterns = {
    "registers": re.compile(r"\\bREG(?:ISTERS)?\\s*[:=]\\s*(\\d+)", re.I),
    "shared": re.compile(r"\\bSHARED\\s*[:=]\\s*(\\d+)", re.I),
    "local": re.compile(r"\\bLOCAL\\s*[:=]\\s*(\\d+)", re.I),
    "stack": re.compile(r"\\bSTACK\\s*[:=]\\s*(\\d+)", re.I),
}

rows = []
for chunk in resource_chunks:
    text = " ".join(chunk)
    name = next((line.strip() for line in chunk
                 if "cryptonight" in line.lower() or "ttable4" in line.lower()), "?")
    fields = {}
    for key, pattern in field_patterns.items():
        m = pattern.search(text)
        if m:
            fields[key] = int(m.group(1))
    if fields:
        rows.append((name, fields))

if rows:
    for name, fields in rows:
        out.append(f"  {name}")
        out.append("    " + " | ".join(
            f"{key}={fields[key]}" for key in ("registers", "shared", "local", "stack")
            if key in fields))
        regs = fields.get("registers", 0)
        local = fields.get("local", 0)
        stack = fields.get("stack", 0)
        if local or stack:
            out.append("    WARNING: non-zero local/stack storage can indicate register spills.")
        if regs >= 128:
            out.append("    NOTE: very high register count; occupancy pressure is likely worth testing.")
        elif regs >= 96:
            out.append("    NOTE: elevated register count; occupancy may be constrained on some GPUs.")
else:
    out.append("  No normalized resource fields parsed; use the raw excerpts above.")

out.append("")
out.append("Interpretation guide")
out.append("-" * 72)
out.append("  1. Non-zero LOCAL/STACK is the first spill signal to attack.")
out.append("  2. If spill-free, compare register counts across ttable/ttable4/cg kernels.")
out.append("  3. Large instruction-count gaps matter only if production selects that kernel.")
out.append("  4. Do not lower registers blindly; validate whole-rotation H/s after each change.")

if not selected:
    out.append("")
    out.append("WARNING: no phase-2 function names matched.")
    out.append("Inspect the full SASS and adjust the function-name needles if nvcc")
    out.append("emitted a different mangled spelling.")

out_path.write_text("\n".join(out) + "\n")
print("\n".join(out))
PY

echo
echo "Filtered SASS:  $SASS"
echo "Resource usage: $RES"
echo "Summary:        $SUMMARY"
echo
echo "Note: the full multi-architecture SASS dump is intentionally not saved."
