# Yerbas Miner 1.0.0.7

Current stable release. Source/tag: `19abf4fcfaff3531fbb34eb8c57bb00d92616092`.

## Changes

- Accelerate custom CPU CryptoNight 1/2/4-way multiplication with an exact native-wide product (`__uint128_t` or MSVC x64 `_umul128`) and a portable fallback. Existing runtime AES dispatch and consensus reference are preserved.
- Extend correctness coverage to 100 arithmetic edge pairs, one million deterministic products, and optimized full GhostRider hashes across the 96-header reference corpus. Offline grouped 1/2/4-way parity and nonce coverage passed across all 20 CN combinations.
- Add explicit offline CPU stage and full-GhostRider plan benchmarks.
- Limit nested NVCC architecture compilation to one worker by default; `YERBAS_NVCC_THREADS` remains configurable.
- Keep generic CPU builds and CUDA targets `52-virtual;61-real;75-real;86-real;89-real;90-real`. No build is specialized to the test machine.

## Overnight evidence

On the test i7-8700K (six CPU workers) and two GTX 1080 Ti cards, the reviewed snapshot of `yerbas-miner-20261015-221717.log` covers 509 completed rotations / 17.65 sampled hours, with 10,344 accepted shares and three stale-job rejects (99.97% accepted). No invalid-hash rejects or CUDA failures were reported. Raw duration-weighted throughput was 1949.84 H/s total: 521.04 CPU and 1428.80 GPU.

For rotations lasting at least 30 seconds, reweighting the earlier 1.0.0.6 run to the new run's distribution of all 20 unordered CN combinations gives:

| Source | Reweighted 1.0.0.6 | 1.0.0.7 | Change |
|---|---:|---:|---:|
| CPU | 414.35 H/s | 517.53 H/s | +24.9% |
| GPU | 1402.81 H/s | 1435.91 H/s | +2.4% |
| Total | 1817.16 H/s | 1953.43 H/s | +7.5% / +136.27 H/s |

All 20 CPU combination averages improved. These are separate runs with different jobs/core schedules, not a randomized experiment or pool-effective accepted-difficulty comparison. GPU kernels were unchanged; do not attribute the GPU difference to the CPU multiply change. Rates are observations on one machine, not guaranteed performance on other hardware.

## Downloads and provenance

- `yerbas-miner-linux-cuda-x86_64.zip`: Linux CUDA build from successful CI run 38842231591.
- `yerbas-miner-windows-cuda-x86_64.zip`: Windows CUDA build from successful CI run 38842231549.
- `yerbas-miner-linux-arm64.zip`: CPU-only ARM64 build and correctness tests from successful CI run 38842231515. It has no NVIDIA CUDA backend.
- `SHA256SUMS.txt`: checksums for all three archives.

Every run is pinned to the source commit above. The publisher verifies successful CI provenance and the embedded source/version strings, repackages the unchanged executables with current release documentation, and includes `BUILD-INFO.txt`. Windows and Linux CPU correctness jobs also passed in run 38842231515. CI is not GPU mining validation on every listed architecture; live hybrid validation was on the test machine above.

## Run

Preserve your existing config and tuning caches when upgrading. Normal GPU tuning should remain `auto`; no fresh retune is required for this CPU change. The measured control used six cached CPU workers and blocking CUDA completion waits, with growth, overlap, the latency controller, combo-policy overrides and profiling disabled.

Linux example (from the extracted directory):

```bash
env -u YERBAS_CUDA_PROFILE -u YERBAS_DIAGNOSTICS \
    -u YERBAS_FAST_LITE_GROWTH -u YERBAS_GPU_LATENCY_TARGET_MS \
    -u YERBAS_CUDA_OVERLAP -u YERBAS_CPU_COMBO_POLICY_FILE \
    YERBAS_CUDA_BLOCKING_WAIT=1 \
    ./yerbas-miner --tune full --gpu-tune auto
```

PowerShell example:

```powershell
$env:YERBAS_CUDA_BLOCKING_WAIT = '1'
.\yerbas-miner.exe --tune full --gpu-tune auto
```

Blocking waits are opt-in, not a changed default. Batch growth and overlap remain experimental/off. Developer-fee behavior, consensus rules, configuration format and garden artwork are unchanged.
