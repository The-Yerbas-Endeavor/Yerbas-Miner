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

## Configuration

Copy the example configuration before running the miner:

```bash
cp config.example.json config.json
```

On Windows, copy `config.example.json` to `config.json` in the same directory as `yerbas-miner.exe`.

Example:

```json
{
  "pool": {
    "url": "stratum+tcp://pool.example.com:3032",
    "user": "YOUR_YERB_ADDRESS",
    "password": "x"
  },
  "miner": {
    "worker": "rig1",
    "threads": 0
  },
  "gpu": {
    "enabled": true,
    "devices": [0],
    "intensity": 0
  },
  "logging": {
    "level": "info"
  }
}
```

`config.json` is ignored by Git so local wallet/pool information is not accidentally committed. `config.example.json` is included in release artifacts.

Command-line options override values from the configuration file:

```bash
./yerbas-miner --pool stratum+tcp://pool.example.com:3032 --user YOUR_YERB_ADDRESS --worker rig1
```

Use a different config file with:

```bash
./yerbas-miner --config myrig.json
```

Common options:

```text
--config FILE
--pool URL
--user USER
--password PASS
--worker NAME
--threads N
--devices 0,1
--intensity N
--no-gpu
--log-level LEVEL
--help
```

Priority is command line, then JSON config, then built-in defaults.

## Current status

- CMake C/C++17 project
- Exact Yerbas Core GhostRider CPU reference path
- Yerbas Core source revision pinned for reproducibility
- 15 SPHlib core hashes wired in
- 6 selectable CryptoNight variants wired in
- Real Yerbas mainnet genesis header test fixture
- Optional NVIDIA CUDA backend
- CUDA GPU discovery and device reporting
- JSON config file support
- command-line configuration overrides
- pool URL/user/password/worker settings
- GPU device/intensity settings
- Stratum endpoint parsing and configuration plumbing
- Stratum socket/session protocol and share submission not implemented yet

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

The first configure requires network access so CMake can fetch the pinned Yerbas Core source tree and the header-only JSON dependency.

## Next milestone

Complete the Stratum TCP session: connect, subscribe, authorize, receive jobs, construct Yerbas block headers, scan nonces, validate shares with the CPU reference, and submit accepted shares. The same real block-header vectors remain the correctness gate for the CUDA implementation.
