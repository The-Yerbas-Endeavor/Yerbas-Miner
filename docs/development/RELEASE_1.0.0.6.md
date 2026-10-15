# Yerbas Miner 1.0.0.6

## Scope and production recommendation

This version adds two explicit performance controls and makes ordinary CMake and development-script CUDA defaults portable. CPU-native compilation remains off by default. Existing tuning profiles and historical logs are preserved.

Recommended measured configuration: cached full CPU policy, cache-first GPU tuning, **blocking CUDA completion waits enabled**, Fast/Lite growth disabled, stagger/overlap disabled. No kernel rewrite or automatic batch-policy promotion is included.

## Blocking CUDA completion waits

Set `YERBAS_CUDA_BLOCKING_WAIT=1` to replace the production scan's stream synchronization with a reusable event created using `cudaEventBlockingSync | cudaEventDisableTiming`. The event is recorded after both asynchronous result copies, and the host waits for completion before reading either result buffer. This lets the host thread sleep while the GPU completes its work.

- Startup reports `scan completion wait | mode=blocking-event` per GPU.
- Unset or any value other than `1` preserves the prior `stream-default` wait.
- Raw-batch and stage-local benchmark modes keep the prior wait behavior.
- Kernels, tuning preferences and scratchpad allocations are unchanged by this option.

The long-run blocking-only reference recorded 321 completed rotation samples over 11.176 sampled hours: **1864.90 H/s total, 1423.08 GPU, 441.82 CPU**. This restores the earlier ~1.86-kH/s performance class. It is not an isolated measurement of the event-wait option's causal gain.

## Experimental Fast/Lite stable-job growth

Set `YERBAS_FAST_LITE_GROWTH=1` only for explicit testing. It remains off by default and is not recommended over the blocking-only control on current evidence.

For each GPU and each newly uploaded job:

1. Start with the ordinary cached/adaptive production batch.
2. Require both CN-Fast and CN-Lite, at least three clean full-size completed scans, and at least 30 seconds of job age.
3. Attempt one increase, no more than 25%, aligned to 128 hashes.
4. Clamp against the current job's actual allocated scratchpad bytes, batch capacity and applicable stale cap.
5. Activate runtime selector/geometry state for the actual dispatched count, including subsequent temporary transition shaping.
6. Reset qualification on the next uploaded job; do not persist a larger preferred batch.

The optional `YERBAS_GPU_LATENCY_TARGET_MS` controller disables this separate growth experiment. Raw/stage-local benchmark modes cannot grow through this API. Existing backend selection may create ordinary batch-keyed cache entries; existing cache files are not deleted or invalidated.

On the measured dual-GTX-1080-Ti host, 3584 grew to 4096/4480 because of its existing allocations; those targets are not hardcoded by GPU model. The completed growth run recorded 286 samples over 9.921 sampled hours: **1856.27 total, 1379.97 GPU, 476.30 CPU H/s**. It had 5458 accepted and six stale rejected shares, without observed CUDA/parity failures. Different rotation mixes and background conditions prevent treating the raw difference as a controlled causal result. Growth has **not demonstrated a sustained whole-miner advantage**.

The short A/B/B/A screen was inconclusive: one B leg never exercised growth, the final A leg had no completed samples, and the modes had insufficient matching Fast/Lite work. Earlier historical fixed-work sweeps also failed to establish a larger-batch advantage.

## Generic build and run

Use one existing build-release directory. Do not use Docker or invalidate tuning caches. The explicit CUDA set matches the portable Linux/Windows CI release targets: compute_52 PTX plus sm_61/sm_75/sm_86/sm_89/sm_90 cubins. A narrower or native development target is an explicit override, not the default. Existing CMake caches retain old values unless overridden, so the canonical command sets both portability options explicitly.

```bash
cd ~/Yerbas-Miner &&
git pull --ff-only origin main &&
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release \
  -DYERBAS_NATIVE_CPU=OFF -DBUILD_TESTING=ON \
  '-DYERBAS_CUDA_ARCHITECTURES=52-virtual;61-real;75-real;86-real;89-real;90-real' &&
cmake --build build-release \
  --target yerbas-miner fast-lite-growth-test ghostrider-smoke ghostrider-equivalence \
  --parallel "$(nproc)" &&
ctest --test-dir build-release --output-on-failure &&
env -u YERBAS_FAST_LITE_GROWTH \
  -u YERBAS_GPU_LATENCY_TARGET_MS \
  -u YERBAS_CUDA_OVERLAP -u YERBAS_DIAGNOSTICS \
  YERBAS_CUDA_BLOCKING_WAIT=1 ./build-release/yerbas-miner --tune full
```

Startup must report version 1.0.0.6, its source commit, the expected CPU/GPU policies and blocking-event waits. Do not set `--gpu-tune full` or CPU retune overrides for ordinary production.

## Validation and interpretation

The growth policy has a CTest covering qualification timing, exclusions, per-job reset, alignment, no-headroom behavior, 25% ceilings, generic capacities and overflow handling. Scheduler C++ syntax and a mock-runtime test of the actual backend method passed during development. Both options were built and exercised on the GPU host using generic targets; the production logs include accepted shares and batch-specific phase parity PASS records. These are not a claim of exhaustive GPU parity coverage on all supported architectures.

CPU reference smoke/equivalence and policy tests are part of the release checks. CI status must be checked on the published commit; no Linux/Windows CI success is asserted by this document itself.

Duration-weighted completed-rotation H/s remains distinct from pool-effective user work. Matching ordered CN triples does not control the other 15 hash stages. Completed stale GPU batches can inflate raw useful-work impressions even with few rejected shares. Compare long uninterrupted runs, accepted difficulty over exact user-active intervals, stale work, and rotation coverage. One-minute-per-hour developer-fee switching is unchanged.

## Next investigation

A lower-rate blocking-only control and a partial CPU-only sample do not yet establish a new CPU implementation regression. The CPU-only sample used six workers, batch 48, one-way lanes, GPU disabled, and had accepted shares. The XMRig comparator has not produced a verified result in the evidence reviewed. Published 778-H/s CPU and 883-H/s GPU listings do not define a same-work six-thread hybrid standard.

Preserve the measured baseline, complete an independent CPU/GPU comparison, and then choose one structural hypothesis. Do not repeat rejected multiply, interleave, stagger, table-ordering, or stage-local batching experiments without new evidence. No sustained hash-rate increase above the recovered baseline is claimed for 1.0.0.6.
