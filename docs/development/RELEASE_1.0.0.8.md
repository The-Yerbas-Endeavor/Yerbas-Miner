# Yerbas Miner 1.0.0.8

## Mining and reliability

- Process GPU completions and restart scans while CPU hashing continues, reducing coordinator idle time in hybrid mining.
- Let CPU workers share batch work while preserving nonce coverage, runtime CPU dispatch and existing hash implementations.
- Count only hashes actually completed during cancellation and shutdown; avoid inflated shutdown hash rates.
- Reuse compatible GPU tuning-cache entries within the current memory budget.
- Improve Linux scratchpad alignment with existing allocation fallbacks.

## Console dashboard

- Consolidate CPU and GPU contributions into one stacked total hash-rate chart.
- Add a worker color key, configurable palettes, run time and rotation information while preserving activity visibility.
- Show 10-minute, 30-minute, 1-hour, 6-hour and 1-day worker averages. Longer averages appear when enough history is available.
- Add component power, H/s/W efficiency and J/kH energy estimates when sensor readings are available.
- Improve Windows console glyph compatibility and print a visible miner-stopped message after shutdown.

## Downloads and compatibility

- `yerbas-miner-linux-cuda-x86_64.zip`: Linux with CUDA support.
- `yerbas-miner-windows-cuda-x86_64.zip`: Windows with CUDA support.
- `yerbas-miner-linux-arm64.zip`: Linux ARM64, CPU-only.
- `SHA256SUMS.txt`: archive checksums.

Distributable builds retain portable CPU settings, runtime feature detection and generic CUDA architecture coverage. Consensus hashing and developer-fee behavior are unchanged. Packages include build provenance and an example configuration; personal configuration, logs and tuning caches are not included.

## Upgrade

Preserve your configuration and tuning caches. Replace the executable and use your existing settings. GPU tuning mode `auto` can reuse compatible cached results. No forced retune is required.

Power estimates use available CPU/GPU telemetry, not wall-socket power. Missing readings are shown as unavailable. Hash-rate results vary with workload and system conditions; this release does not promise a fixed speedup.

## Validation

Correctness checks cover GhostRider reference parity, nonce coverage, partial cancellation, callback failure recovery, cache limits, rolling averages and efficiency arithmetic. Release packages are published only after the release workflow's required platform builds and checks succeed.
