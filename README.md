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
    "cpu_enabled": true,
    "threads": 0,
    "cpu_batch": 0,
    "cpu_tune": "off"
  },
  "gpu": {
    "enabled": true,
    "devices": [0],
    "intensity": 0,
    "gpu_tune": "auto",
    "skip_validation": false
  },
  "logging": {
    "level": "info",
    "console": "auto"
  }
}
```

`config.json` is ignored by Git so local wallet/pool information is not accidentally committed. `config.example.json` is included in release artifacts.

### Console modes

Yerbas-Miner now uses a single-screen mining dashboard automatically when stdout is an interactive terminal and the session log is active. Detailed mining events continue to the session log while the terminal is refreshed in place.

```text
auto   default; use the TUI on an interactive terminal and plain output otherwise
tui    request the single-screen dashboard
plain  traditional scrolling console
```

The default session log is written under `logs/`. Use `--console plain` for redirected output, troubleshooting, or the traditional scrolling view. `--log-level debug` keeps the plain diagnostic console behavior when console mode is `auto`.


### CPU tuning modes

CPU tuning is optional. The default example config uses `"cpu_tune": "default"`; set it to `"off"` when immediate startup with the configured/default CPU thread count and batch size is preferred.

Available modes:

```text
off      no CPU benchmark; start mining immediately
simple   quick thread/batch production tuning
default  balanced thread/batch production tuning
full     exhaustive thread/batch tuning and future full GhostRider rotation tuning
```

Command-line equivalents:

```bash
./yerbas-miner --no-tune
./yerbas-miner --tune simple
./yerbas-miner --tune default
./yerbas-miner --tune full
```

`threads` is a ceiling when tuning is enabled. `threads: 0` allows the tuner to use all logical CPUs; for example, `threads: 6` on a 12-thread machine limits the search to configurations using at most 6 CPU workers.

The current production tuner selects CPU thread count and per-thread batch size by measuring full GhostRider throughput across representative schedules. True cpuminer-gr-style per-rotation `1way/2way/4way` CryptoNight selection requires genuine multi-lane CryptoNight kernels. Yerbas-Miner currently has experimental 2-way/4-way work behind parity and production-performance gates; wider execution is not selected unless it proves safe and faster than the 1-way production path.

### GPU tuning modes

`gpu_tune` provides the normal user-facing CUDA tuning policy. The recommended and shipped default is `"auto"`.

```text
auto   cache-first production mode. Saved hardware/class, per-variant batch,
       phase backend, kernel-selector and geometry tuning is reused immediately.
       On a new interactive machine with no GPU calibration profile, the normal
       first-run flow can perform one bounded hardware calibration and save it.

off    do not request GPU autotuning. Existing valid caches may still be reused;
       otherwise the CUDA backend uses safe runtime-derived production settings.

full   deliberately rebuild GPU tuning. Runs fresh bounded class/per-variant
       calibration and enables the deep production selector/geometry retune as
       real GhostRider rotations and batch sizes are encountered. This mode can
       take a long time and is intended for deliberate retuning, not every start.
```

Use `full` after a meaningful hardware/driver/toolkit change, when validating a new CUDA optimization, or when intentionally rebuilding the tuning caches. Normal mining should use `auto`.

Command-line equivalent:

```bash
./yerbas-miner --gpu-tune auto
./yerbas-miner --gpu-tune full
./yerbas-miner --gpu-tune off
```

The older `--gpu-autotune` option remains available as a bounded one-shot GPU calibration and does not implicitly enable the long deep production retune. Developer environment variables such as `YERBAS_GPU_AUTOTUNE`, `YERBAS_GPU_VARIANT_AUTOTUNE`, and `YERBAS_CUDA_RETUNE` remain available as explicit overrides for development/testing.

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
--cpu-batch N
--tune off|simple|default|full
--no-tune
--devices 0,1
--intensity N
--gpu-tune off|auto|full
--gpu-autotune
--no-gpu
--skip-validation
--log-level LEVEL
--help
```

Priority is command line, then JSON config, then built-in defaults. Explicit developer tuning environment variables remain authoritative when they are set.

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
- optional CPU production autotuning with direct no-tune startup
- cache-first GPU production tuning with `auto`, `full`, and `off` modes
- pool URL/user/password/worker settings
- GPU device/intensity settings
- Stratum endpoint parsing and configuration plumbing

## Build

### Fast development/testing build

For normal development and repeated testing, use:

```bash
bash scripts/configure-dev.sh
```

This creates `build-dev/yerbas-miner` with:

- `YERBAS_NATIVE_CPU=OFF`, so CPU code remains generic and is not compiled with `-march=native` / `-mtune=native`.
- CUDA architecture `native` by default, so only the GPU architecture present on the development machine is compiled instead of every supported release architecture.
- The same runtime CPU dispatch, CPU/GPU tuning, mining logic, and validation paths used by the release build.
- A separate `build-dev` directory, so incremental development rebuilds do not disturb the full release build tree.

Run it with:

```bash
./build-dev/yerbas-miner
```

For repeated development after the first configure, keep the same build directory:

```bash
git pull
bash scripts/configure-dev.sh
```

The script reuses `build-dev` and Ninja only recompiles files affected by the changes.

To override the development CUDA target manually:

```bash
YERBAS_DEV_CUDA_ARCHITECTURES=61 bash scripts/configure-dev.sh
```

Use this only when you intentionally want a specific development GPU architecture instead of automatic native detection.

### Full generic release build

Use the full release build only when validating or producing a distributable binary:

```bash
bash scripts/configure-release.sh
```

This creates `build-release/yerbas-miner` with:

- `YERBAS_NATIVE_CPU=OFF`, keeping CPU code portable.
- A CUDA fat binary for compute capabilities `52;60;61;70;75;80;86;89;90`.
- The same runtime CPU/GPU autotuning and production code used by the development build.

Run it with:

```bash
./build-release/yerbas-miner
```

The full release build is intentionally slower because CUDA code is generated and linked for every supported architecture. Do not use it as the normal edit/test loop.

### Optional: native CPU benchmark build

For local CPU performance experiments only:

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
