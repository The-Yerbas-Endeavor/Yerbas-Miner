#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build-release}"

# CUDA 12.4 on current Ubuntu releases is commonly paired with GCC 13.
# Keep C, C++, CUDA host compilation, and final C++ linking on the same
# libstdc++ ABI to avoid mixing newer GCC objects with an older CUDA runtime
# search path (for example unresolved __cxa_call_terminate at final link).
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

# CMake does not safely change compilers inside an already-configured build
# tree. If this build directory came from another compiler, discard only the
# generated CMake/Ninja configuration and let dependencies/artifacts be rebuilt
# consistently by the selected toolchain.
if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  OLD_CXX="$(sed -n 's/^CMAKE_CXX_COMPILER:FILEPATH=//p' "${BUILD_DIR}/CMakeCache.txt" | head -n1)"
  if [[ -z "${OLD_CXX}" ]]; then
    OLD_CXX="$(sed -n 's/^CMAKE_CXX_COMPILER:STRING=//p' "${BUILD_DIR}/CMakeCache.txt" | head -n1)"
  fi
  if [[ -n "${OLD_CXX}" ]]; then
    OLD_CXX="$(readlink -f "${OLD_CXX}" 2>/dev/null || printf '%s' "${OLD_CXX}")"
    if [[ "${OLD_CXX}" != "${CXX_BIN}" ]]; then
      echo "Release toolchain changed: ${OLD_CXX} -> ${CXX_BIN}"
      echo "Resetting ${BUILD_DIR} CMake configuration to prevent mixed C++ ABI objects."
      rm -rf "${BUILD_DIR}/CMakeCache.txt" "${BUILD_DIR}/CMakeFiles"
    fi
  fi
fi

echo "Release host toolchain:"
echo "  CC=${CC_BIN}"
echo "  CXX=${CXX_BIN}"
echo "  CUDA host CXX=${CXX_BIN}"

cmake -S . -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${CC_BIN}" \
  -DCMAKE_CXX_COMPILER="${CXX_BIN}" \
  -DCMAKE_CUDA_HOST_COMPILER="${CXX_BIN}" \
  -DYERBAS_ENABLE_CUDA=ON \
  -DYERBAS_NATIVE_CPU=OFF \
  -DYERBAS_CUDA_ARCHITECTURES="52;60;61;70;75;80;86;89;90"

cmake --build "${BUILD_DIR}" \
  --target yerbas-miner \
  --parallel "$(nproc)"

echo
echo "Generic portable Yerbas Miner built in ${BUILD_DIR}."
echo "CPU code is not compiled with -march=native/-mtune=native."
echo "CUDA fat binary includes supported NVIDIA architectures: 52,60,61,70,75,80,86,89,90."
echo "Runtime CPU/GPU autotuning will select production settings on the target machine."
