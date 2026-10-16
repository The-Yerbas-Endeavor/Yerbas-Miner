# Yerbas Miner 1.0.0.7

Current stable release. Source/tag: `19abf4fcfaff3531fbb34eb8c57bb00d92616092`.

## Changes

- Improve CPU CryptoNight performance with exact native-width multiplication and a portable fallback.
- Preserve runtime CPU feature detection and the pinned consensus reference.
- Expand arithmetic and optimized GhostRider correctness checks.
- Add offline CPU stage and full-GhostRider benchmarks.
- Reduce default nested CUDA compiler parallelism, with an explicit build-time override.
- Keep distributable builds portable across supported systems.

## Validation

Arithmetic, reference-equivalence, grouped execution and nonce-coverage checks passed. The release also completed extended mining validation. Successful CI builds and correctness jobs are pinned to the source commit above.

## Downloads

- `yerbas-miner-linux-cuda-x86_64.zip`: Linux with CUDA support.
- `yerbas-miner-windows-cuda-x86_64.zip`: Windows with CUDA support.
- `yerbas-miner-linux-arm64.zip`: Linux ARM64, CPU-only.
- `SHA256SUMS.txt`: checksums for all three archives.

Packages contain unchanged verified executables, release documentation and `BUILD-INFO.txt`. The publisher verifies source/version metadata and successful CI provenance before uploading the archives.

## Upgrade and run

Preserve your existing configuration and tuning caches when upgrading. Use GPU tuning mode `auto`; this CPU update does not require a fresh GPU retune.

Linux:

```bash
./yerbas-miner --tune full --gpu-tune auto
```

PowerShell:

```powershell
.\yerbas-miner.exe --tune full --gpu-tune auto
```

Blocking CUDA completion waits remain opt-in. Batch growth and overlap remain experimental and off by default. Developer-fee behavior, consensus rules, configuration format and console artwork are unchanged.
