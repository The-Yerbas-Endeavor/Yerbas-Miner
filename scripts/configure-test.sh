#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build-test}"
CUDA_ARCHS="${YERBAS_TEST_CUDA_ARCHS:-61}"

# Fast optimized local test build.
#
# This intentionally keeps Release optimization but limits CUDA code generation
# to the architecture(s) requested through YERBAS_TEST_CUDA_ARCHS. It is meant
# for rapid local iteration only. Official/generic builds should continue to use
# scripts/configure-release.sh, which produces the full multi-architecture fat
# binary.

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

if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  OLD_CXX="$(sed -n 's/^CMAKE_CXX_COMPILER:FILEPATH=//p' "${BUILD_DIR}/CMakeCache.txt" | head -n1)"
  if [[ -z "${OLD_CXX}" ]]; then
    OLD_CXX="$(sed -n 's/^CMAKE_CXX_COMPILER:STRING=//p' "${BUILD_DIR}/CMakeCache.txt" | head -n1)"
  fi
  if [[ -n "${OLD_CXX}" ]]; then
    OLD_CXX="$(readlink -f "${OLD_CXX}" 2>/dev/null || printf '%s' "${OLD_CXX}")"
    if [[ "${OLD_CXX}" != "${CXX_BIN}" ]]; then
      echo "Test toolchain changed: ${OLD_CXX} -> ${CXX_BIN}"
      echo "Resetting ${BUILD_DIR} CMake configuration to prevent mixed C++ ABI objects."
      rm -rf "${BUILD_DIR}/CMakeCache.txt" "${BUILD_DIR}/CMakeFiles"
    fi
  fi
fi

echo "Fast Release test toolchain:"
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
echo "Fast optimized Yerbas Miner test build ready in ${BUILD_DIR}."
echo "CUDA targets: ${CUDA_ARCHS}"
echo "Do not delete ${BUILD_DIR} between ordinary source changes; rerun this script or use:"
echo "  cmake --build ${BUILD_DIR} --target yerbas-miner --parallel \"\$(nproc)\""
echo
echo "For a generic release build, use:"
echo "  bash scripts/configure-release.sh"
