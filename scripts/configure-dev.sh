#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build-dev}"
CUDA_ARCHS="${YERBAS_DEV_CUDA_ARCHITECTURES:-native}"

# Keep the development build portable on the CPU. Only the CUDA target is
# narrowed to the GPU(s) present on the development machine so iterative
# builds do not rebuild every release architecture.
CC_BIN="${YERBAS_CC:-}"
CXX_BIN="${YERBAS_CXX:-}"

if [[ -z "${CC_BIN}" || -z "${CXX_BIN}" ]]; then
  if command -v gcc-13 >/dev/null 2>&1 && command -v g++-13 >/dev/null 2>&1; then
    CC_BIN="$(command -v gcc-13)"
    CXX_BIN="$(command -v g++-13)"
  else
    CC_BIN="${CC_BIN:-$(command -v gcc)}"
    CXX_BIN="${CXX_BIN:-$(command -v g++)}"
  fi
fi

CC_BIN="$(readlink -f "${CC_BIN}")"
CXX_BIN="$(readlink -f "${CXX_BIN}")"

# A CMake tree cannot safely switch host compilers or CUDA architecture sets
# in place. Reset only generated configuration when either setting changed.
RESET=0
if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  OLD_CXX="$(sed -n 's/^CMAKE_CXX_COMPILER:FILEPATH=//p' "${BUILD_DIR}/CMakeCache.txt" | head -n1)"
  if [[ -z "${OLD_CXX}" ]]; then
    OLD_CXX="$(sed -n 's/^CMAKE_CXX_COMPILER:STRING=//p' "${BUILD_DIR}/CMakeCache.txt" | head -n1)"
  fi
  if [[ -n "${OLD_CXX}" ]]; then
    OLD_CXX="$(readlink -f "${OLD_CXX}" 2>/dev/null || printf '%s' "${OLD_CXX}")"
    [[ "${OLD_CXX}" != "${CXX_BIN}" ]] && RESET=1
  fi

  OLD_ARCHS="$(sed -n 's/^CMAKE_CUDA_ARCHITECTURES:STRING=//p' "${BUILD_DIR}/CMakeCache.txt" | head -n1)"
  if [[ -n "${OLD_ARCHS}" && "${OLD_ARCHS}" != "${CUDA_ARCHS}" ]]; then
    RESET=1
  fi
fi

if [[ "${RESET}" == "1" ]]; then
  echo "Development build configuration changed; resetting ${BUILD_DIR} CMake state."
  rm -rf "${BUILD_DIR}/CMakeCache.txt" "${BUILD_DIR}/CMakeFiles"
fi

echo "Development host toolchain:"
echo "  CC=${CC_BIN}"
echo "  CXX=${CXX_BIN}"
echo "  CUDA host CXX=${CXX_BIN}"
echo "  CUDA architectures=${CUDA_ARCHS}"

cmake -S . -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${CC_BIN}" \
  -DCMAKE_CXX_COMPILER="${CXX_BIN}" \
  -DCMAKE_CUDA_HOST_COMPILER="${CXX_BIN}" \
  -DYERBAS_ENABLE_CUDA=ON \
  -DYERBAS_NATIVE_CPU=OFF \
  -DYERBAS_CUDA_ARCHITECTURES="${CUDA_ARCHS}"

cmake --build "${BUILD_DIR}" \
  --target yerbas-miner \
  --parallel "$(nproc)"

echo
echo "Development Yerbas Miner built in ${BUILD_DIR}."
echo "CPU remains generic/portable (no -march=native or -mtune=native)."
echo "CUDA targets only: ${CUDA_ARCHS}."
echo "Use scripts/configure-release.sh only for the full distributable fat binary."
