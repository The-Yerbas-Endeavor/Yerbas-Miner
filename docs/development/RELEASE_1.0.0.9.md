# Yerbas Miner 1.0.0.9

## CPU work scheduling

- Refill shared CPU work in smaller groups so available workers can pick up remaining hashes sooner.
- Preserve the minimum group required by the selected lane width, existing hash kernels, runtime CPU feature detection and GPU scheduling.
- Retain normal autotuning and compatible tuning caches. Worker counts, batch sizes and affinity remain selected for the running system; no machine-specific policy is bundled.

## Downloads

- `yerbas-miner-windows-cuda-x86_64.zip`: Windows x86_64 with CUDA support.
- `yerbas-miner-linux-cuda-x86_64.zip`: Linux x86_64 with CUDA support.
- `yerbas-miner-linux-arm64.zip`: Linux ARM64, CPU-only.
- `SHA256SUMS.txt`: checksums for all release archives.

CPU builds remain portable with runtime dispatch. CUDA packages retain generic architecture coverage. Packages contain an example configuration and build provenance; personal configurations, logs and tuning caches are excluded.

## Upgrade and tuning

Preserve your configuration and tuning caches when replacing the executable. Normal starts can reuse compatible tuning results. To request fresh CPU and bounded GPU calibration, run:

```text
yerbas-miner --autotune --tune full --gpu-tune auto
```

On Windows, use `yerbas-miner.exe`. Let calibration finish with other compute-heavy workloads stopped. Omit `--autotune` on subsequent starts to reuse the selected policy. Deep GPU retuning remains separately available through `--gpu-tune full`.

## Validation

The release checks include reference parity across all CryptoNight combinations, shared-work nonce coverage with odd batch tails, cancellation, callback failure recovery and existing dashboard/cache checks. Windows CUDA, Linux CUDA and Linux ARM64 packages are built from the same release commit. Publication is gated on required builds and correctness checks.

Controlled fixed-input CPU comparisons support the scheduling change; end-to-end gains depend on rotation mix, tuning and competing workloads. This release does not promise a fixed hashrate or speedup. Consensus hashing, developer-fee behavior and existing dashboard features are unchanged.
