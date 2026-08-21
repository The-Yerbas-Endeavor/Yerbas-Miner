#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BUILD_DIR="build-cuda"
CORE_DIR="$BUILD_DIR/_deps/yerbas_core-src"
SLOW_HASH="$CORE_DIR/src/cryptonote/slow-hash.c"

# Configure only when the build tree does not exist. CMake installs the base
# validator instrumentation into the fetched Core slow-hash.c.
if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  cmake -S . -B "$BUILD_DIR"
fi

if [[ ! -f "$SLOW_HASH" ]]; then
  echo "Missing fetched Yerbas Core source; configuring build tree..."
  cmake -S . -B "$BUILD_DIR"
fi

# If the detailed trace from a previous run is still present, keep it. If the
# base CMake hook is missing entirely, reconfigure once to restore it.
if grep -q "YERBAS_CN_FIRST_ITER_TRACE_V2" "$SLOW_HASH"; then
  echo "Core first-iteration trace already installed"
elif grep -q "yerbas_cn_debug_get_last_state" "$SLOW_HASH"; then
  python3 tools/trace_core_cn.py "$SLOW_HASH"
else
  echo "Restoring validator instrumentation with CMake..."
  cmake -S . -B "$BUILD_DIR"
  python3 tools/trace_core_cn.py "$SLOW_HASH"
fi

# Ninja handles file-level parallelism with $(nproc). NVCC is configured with
# --threads=0 for its own supported internal parallel work.
cmake --build "$BUILD_DIR" \
  --target cuda-keccak-validation \
  --parallel "$(nproc)"

./"$BUILD_DIR"/cuda-keccak-validation
