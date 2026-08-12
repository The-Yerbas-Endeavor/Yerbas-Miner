# Yerbas-Miner

Yerbas-Miner is the GPU-miner development repository for the Yerbas network.

The current `main` branch contains the initial C++/CUDA scaffold for a GhostRider miner. It is intentionally conservative: the mining loop remains disabled until the exact Yerbas-compatible GhostRider CPU reference implementation and test vectors are imported and verified.

## Current status

- CMake C++17 project
- Optional NVIDIA CUDA backend
- CUDA GPU discovery and device reporting
- GhostRider CPU reference interface
- Smoke-test framework
- Safe refusal to mine with an unverified hash implementation

## Build

CPU-only scaffold:

```bash
cmake -S . -B build -DYERBAS_ENABLE_CUDA=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/yerbas-miner
```

CUDA build (requires the NVIDIA CUDA Toolkit):

```bash
cmake -S . -B build -DYERBAS_ENABLE_CUDA=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/yerbas-miner
```

## Next milestone

Import the exact Yerbas GhostRider reference implementation and deterministic test vectors, validate CPU hashes, then add CUDA kernels incrementally and require CPU/GPU equality before enabling nonce scanning or Stratum share submission.
