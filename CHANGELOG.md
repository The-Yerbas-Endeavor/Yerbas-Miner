# Changelog

## 1.0.0.8 — 2026-10-19

- Service GPU completions during CPU batches and share CPU batch work across active workers.
- Preserve actual completed-hash accounting through cancellation and shutdown.
- Adapt compatible GPU tuning-cache entries to current memory budgets and improve Linux scratchpad alignment.
- Consolidate hash-rate charts, add configurable worker colors, rolling averages and component power-efficiency estimates.
- Improve console compatibility and show a visible stopped message.
- Retain generic CPU/CUDA builds and publish checked platform archives with checksums.

See [1.0.0.8 release notes](docs/development/RELEASE_1.0.0.8.md).

## 1.0.0.7 — 2026-10-16

- Use exact native-width CPU CN multiplication where supported, with a portable fallback and unchanged runtime AES dispatch. Add carry/overflow checks and optimized full-GhostRider equivalence coverage. Offline screens and the reviewed 17.65-hour hybrid run support a sustained CPU gain. See the release notes for workload-adjusted results and attribution limits.

- Add an offline full-GhostRider worker/width benchmark across all 20 CN combinations with grouped reference parity and nonce-coverage checks; document rejected scalar-state, fewer-worker and wider-policy screens. Worker/width defaults remain unchanged.

- Preserve 1.0.0.6 (`30a1d5a06b9c`) as the stable performance baseline.
- Add an explicit offline, single-worker CPU CN benchmark with reference parity checks for all six variants; no pool connection or cache writes.
- Limit nested NVCC architecture compilation to one worker by default, with an explicit `YERBAS_NVCC_THREADS` override. This changes build resource usage, not mining code or hardware portability.
- Reject early CPU prefetch, scalar accumulator and global wider-policy candidates; release the validated native-width multiply change.
- Publish generic Linux/Windows CUDA and CPU-only ARM64 binaries from successful CI on `19abf4fcfaff`, with checksums and build provenance.

See [1.0.0.7 release notes](docs/development/RELEASE_1.0.0.7.md) for overnight evidence and run instructions.

## 1.0.0.6

- Render the same full-detail Braille garden by default on Windows and Linux; retain explicit block/ASCII font compatibility modes.

- Add opt-in blocking CUDA scan completion waits using a reusable blocking event after result readbacks. Existing stream synchronization remains the default unless `YERBAS_CUDA_BLOCKING_WAIT=1` is set.
- Add opt-in, memory-bounded Fast/Lite stable-job batch growth, with three-clean-scan/30-second qualification and a 25% ceiling. It remains experimental and off by default; long runs have not demonstrated an overall advantage.
- Track per-job qualification and activate batch-specific runtime state when the experiment changes scan sizes.
- Add growth-policy boundary and lifecycle tests.
- Use portable CUDA PTX/cubin defaults in ordinary CMake, development and release configuration; CPU-native compilation remains disabled by default.
- Document the recovered ~1.86-kH/s class baseline, inconclusive growth evidence, generic build/run command, and remaining CPU/GPU benchmark work.

See [detailed release notes](docs/development/RELEASE_1.0.0.6.md). Tuning caches, existing logs, consensus kernels, developer-fee behavior and automatic stagger disabling are preserved.
