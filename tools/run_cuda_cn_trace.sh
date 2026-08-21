#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BUILD_DIR="build-cuda"
CORE_DIR="$BUILD_DIR/_deps/yerbas_core-src"
SLOW_HASH="$CORE_DIR/src/cryptonote/slow-hash.c"

# Configure only when the build tree does not exist. CMake installs the base
# validator instrumentation into the fetched Core slow-hash.c. Do NOT git
# checkout that file here: doing so removes the instrumentation that
# trace_core_cn.py extends.
if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
  cmake -S . -B "$BUILD_DIR"
fi

if [[ ! -f "$SLOW_HASH" ]]; then
  echo "Missing fetched Yerbas Core source; configuring build tree..."
  cmake -S . -B "$BUILD_DIR"
fi

# If the base hook is absent (for example after a manual git checkout inside
# _deps), repair it automatically with one configure instead of making the
# user diagnose the build tree.
if ! grep -q "yerbas_cn_debug_get_last_state" "$SLOW_HASH"; then
  echo "Restoring validator instrumentation with CMake..."
  cmake -S . -B "$BUILD_DIR"
fi

python3 tools/trace_core_cn.py "$SLOW_HASH"

# Ninja handles file-level parallelism with $(nproc). NVCC is configured with
# --threads=0 for its own supported internal parallel work.
cmake --build "$BUILD_DIR" \
  --target cuda-keccak-validation \
  --parallel "$(nproc)"

./"$BUILD_DIR"/cuda-keccak-validation
