#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BUILD_DIR="build-cuda"
CORE_DIR="$BUILD_DIR/_deps/yerbas_core-src"
SLOW_HASH="$CORE_DIR/src/cryptonote/slow-hash.c"
TRACE_MARKER="YERBAS_CN_FIRST_ITER_TRACE_V3"

# Configure only when the build tree does not exist. CMake installs the base
# validator instrumentation into the fetched Core slow-hash.c.
if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  cmake -S . -B "$BUILD_DIR"
fi

if [[ ! -f "$SLOW_HASH" ]]; then
  echo "Missing fetched Yerbas Core source; configuring build tree..."
  cmake -S . -B "$BUILD_DIR"
fi

# A previous trace version may still be installed in the fetched Core file.
# Reconfigure once to restore the pinned source plus the base validator hook
# whenever the current trace marker is absent. This keeps trace updates
# deterministic without forcing a full configure on every run.
if ! grep -q "$TRACE_MARKER" "$SLOW_HASH"; then
  echo "Refreshing validator instrumentation for current trace..."
  cmake -S . -B "$BUILD_DIR"
fi

# If the base hook is still absent, fail clearly before attempting injection.
if ! grep -q "yerbas_cn_debug_get_last_state" "$SLOW_HASH"; then
  echo "Validator instrumentation is still missing after configure" >&2
  exit 1
fi

python3 tools/trace_core_cn.py "$SLOW_HASH"

# Ninja handles file-level parallelism with $(nproc). NVCC is configured with
# --threads=0 for its own supported internal parallel work.
cmake --build "$BUILD_DIR" \
  --target cuda-keccak-validation \
  --parallel "$(nproc)"

./"$BUILD_DIR"/cuda-keccak-validation
