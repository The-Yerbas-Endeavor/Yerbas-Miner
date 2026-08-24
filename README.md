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

### Recommended: generic portable CPU + CUDA build

For normal development, testing, and release validation, use the portable release script:

```bash
bash scripts/configure-release.sh
```

This creates `build-release/yerbas-miner` with:

- `YERBAS_NATIVE_CPU=OFF`, so CPU code is not compiled with `-march=native` / `-mtune=native`.
- A multi-architecture CUDA fat binary for compute capabilities `52;60;61;70;75;80;86;89;90`.
- Runtime CPU and GPU autotuning so production settings are selected on the target machine.

Run it with:

```bash
./build-release/yerbas-miner
```

Keep this as the normal development/testing path when validating behavior intended for general users.

### Optional: native benchmark build

For local performance experiments only:

```bash
bash scripts/configure-native-cpu.sh
```

This creates `build-native/yerbas-miner` and intentionally enables host-specific CPU optimization plus native CUDA architecture detection. It may be faster on the machine that compiled it, but it should not be redistributed or used to validate portability.

### CPU-only reference build

```bash
cmake -S . -B build -DYERBAS_ENABLE_CUDA=OFF -DYERBAS_NATIVE_CPU=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/yerbas-miner
```

### Manual generic CUDA build

```bash
cmake -S . -B build-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DYERBAS_ENABLE_CUDA=ON \
  -DYERBAS_NATIVE_CPU=OFF \
  -DYERBAS_CUDA_ARCHITECTURES="52;60;61;70;75;80;86;89;90"
cmake --build build-release --parallel
ctest --test-dir build-release --output-on-failure
./build-release/yerbas-miner
```

For faster repeated portable builds, keep the same `build-release` directory and run only:

```bash
git pull
cmake --build build-release --parallel $(nproc)
```

Optional `ccache` support can be enabled when installed:

```bash
cmake -S . -B build-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DYERBAS_ENABLE_CUDA=ON \
  -DYERBAS_NATIVE_CPU=OFF \
  -DYERBAS_CUDA_ARCHITECTURES="52;60;61;70;75;80;86;89;90" \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
```

The first configure requires network access so CMake can fetch the pinned Yerbas Core source tree and the header-only JSON dependency.

## CPU power and efficiency telemetry on Linux

Yerbas-Miner can display CPU package power and calculate CPU mining efficiency in hashes per watt (`H/W`) when the operating system exposes a readable package-energy or power sensor.

On many Intel Linux systems, package energy is exposed through the kernel's RAPL powercap interface. Some distributions restrict unprivileged access to the RAPL `energy_uj` file. If the status table shows CPU temperature but reports `n/a` for CPU `POWER` and `EFFICIENCY`, check whether the package energy counter is readable:

```bash
cat /sys/devices/virtual/powercap/intel-rapl/intel-rapl:0/energy_uj
```

If this returns `Permission denied`, allow read access with:

```bash
sudo chmod a+r /sys/devices/virtual/powercap/intel-rapl/intel-rapl:0/energy_uj
```

Verify the counter can then be read without `sudo`:

```bash
cat /sys/devices/virtual/powercap/intel-rapl/intel-rapl:0/energy_uj
```

Restart Yerbas-Miner after changing the permission. Once package power is readable, the status table can report CPU watts and calculate CPU `H/W` efficiency.

The exact powercap path can vary by CPU, kernel, and Linux distribution. Yerbas-Miner discovers supported telemetry interfaces at runtime; the path above is the common Intel `package-0` RAPL location. Do not run Yerbas-Miner as root solely for telemetry. Systems that do not expose a readable CPU package-power interface will continue mining normally and display `n/a` for unavailable power and efficiency values.

Note that permissions under `/sys` may be reset after reboot. If persistent CPU power telemetry is desired, configure the appropriate system permission/udev policy for the machine rather than running the miner as root.

## Next milestone

Complete the Stratum TCP session: connect, subscribe, authorize, receive jobs, construct Yerbas block headers, scan nonces, validate shares with the CPU reference, and submit accepted shares. The same real block-header vectors remain the correctness gate for the CUDA implementation.
