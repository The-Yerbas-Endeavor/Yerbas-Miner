# Yerbas-Miner

Yerbas-Miner is the GPU-miner development repository for the Yerbas network.

The current `main` branch contains a C++/CUDA miner scaffold plus a CPU GhostRider reference path wired directly to the implementation used by Yerbas Core.

## GhostRider reference

The build fetches Yerbas Core and pins it to commit:

`040073f22e2b496b21e07eebfc6ca97e22b4cd40`

That Core revision supplies the real `HashSelection`, SPHlib core hashes, and CryptoNight slow-hash implementations used by `CBlockHeader::ComputeHash()`.

Yerbas Core computes PoW as:

```text
HashGR(serialized block header, hashPrevBlock)
```

The miner adapter accepts the serialized block header and derives `hashPrevBlock` directly from header bytes 4-35, matching Core's serialized `CBlockHeader` layout and avoiding display-endian mistakes.

GhostRider then runs 18 stages:

```text
5 core hashes -> CryptoNight -> 5 core hashes -> CryptoNight -> 5 core hashes -> CryptoNight
```

The 15 core hashes and three CryptoNight selections are determined from the previous block hash using Yerbas Core's exact `HashSelection` logic.

## Current status

- CMake C/C++17 project
- Exact Yerbas Core GhostRider CPU reference path
- Yerbas Core source revision pinned for reproducibility
- 15 SPHlib core hashes wired in
- 6 selectable CryptoNight variants wired in
- Optional NVIDIA CUDA backend
- CUDA GPU discovery and device reporting
- CPU GhostRider smoke test
- Mining loop still disabled until deterministic known-good block vectors are added and CPU results are verified against Yerbas Core RPC/block data

## Build

CPU reference build:

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

The first configure requires network access so CMake can fetch the pinned Yerbas Core source tree.

## Next milestone

Add deterministic real Yerbas block-header/PoW test vectors and verify the CPU adapter byte-for-byte against Core. Once those vectors pass, the same vectors become the correctness gate for the CUDA implementation before nonce scanning or share submission is enabled.
