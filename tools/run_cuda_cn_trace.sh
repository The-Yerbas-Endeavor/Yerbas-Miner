#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

restore_validator() {
  git checkout -- tests/cuda_keccak_validation.cpp >/dev/null 2>&1 || true
}
trap restore_validator EXIT

cmake -S . -B build-cuda
python3 tools/trace_core_cn.py build-cuda/_deps/yerbas_core-src/src/cryptonote/slow-hash.c
python3 tools/trace_cuda_cn.py tests/cuda_keccak_validation.cpp
cmake --build build-cuda --target cuda-keccak-validation --parallel "$(nproc)"
./build-cuda/cuda-keccak-validation
