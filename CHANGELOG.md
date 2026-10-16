# Changelog

## 1.0.0.7 — in development

- Use exact native-width CPU CN multiplication where supported, with a portable fallback and unchanged runtime AES dispatch. Add carry/overflow checks and optimized full-GhostRider equivalence coverage. CPU-only screens show a candidate gain; sustained hybrid/pool performance remains to be validated.

- Add an offline full-GhostRider worker/width benchmark across all 20 CN combinations with grouped reference parity and nonce-coverage checks; document rejected scalar-state, fewer-worker and wider-policy screens. Production defaults remain unchanged.

- Preserve 1.0.0.6 (`30a1d5a06b9c`) as the stable performance baseline.
- Add an explicit offline, single-worker CPU CN benchmark with reference parity checks for all six variants; no pool connection or cache writes.
- Limit nested NVCC architecture compilation to one worker by default, with an explicit `YERBAS_NVCC_THREADS` override. This changes build resource usage, not mining code or hardware portability.
- Screen early CPU scratchpad prefetch in A/B/B/A order; reject it because most variants regress. No sustained hash-rate improvement is promoted yet.

## 1.0.0.6

- Render the same full-detail Braille garden by default on Windows and Linux; retain explicit block/ASCII font compatibility modes.

- Add opt-in blocking CUDA scan completion waits using a reusable blocking event after result readbacks. Existing stream synchronization remains the default unless `YERBAS_CUDA_BLOCKING_WAIT=1` is set.
- Add opt-in, memory-bounded Fast/Lite stable-job batch growth, with three-clean-scan/30-second qualification and a 25% ceiling. It remains experimental and off by default; long runs have not demonstrated an overall advantage.
- Track per-job qualification and activate batch-specific runtime state when the experiment changes scan sizes.
- Add growth-policy boundary and lifecycle tests.
- Use portable CUDA PTX/cubin defaults in ordinary CMake, development and release configuration; CPU-native compilation remains disabled by default.
- Document the recovered ~1.86-kH/s class baseline, inconclusive growth evidence, generic build/run command, and remaining CPU/GPU benchmark work.

See [detailed release notes](docs/development/RELEASE_1.0.0.6.md). Tuning caches, existing logs, consensus kernels, developer-fee behavior and automatic stagger disabling are preserved.
