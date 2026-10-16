# 1.0.0.7 performance development — 2026-10-15

## Stable baseline

Current stable source is 1.0.0.6 at `30a1d5a06b9c344354106619afbab980d7573205`.
Keep that source and its tuning caches as the control. The latest supplied run
contains 259 completed rotations over 9.006 sampled hours: duration-weighted
1815.38 H/s total, 1394.39 GPU and 420.99 CPU. It records 5749 accepted shares and
six stale rejects. These raw rates are rotation-dependent and differ from pool
accepted-difficulty work per unit time. The historical ~1865-H/s reference is
also preserved; unlike workloads cannot establish a regression or improvement.

## First screen: early CPU scratchpad prefetch — rejected

A temporary candidate issued a T0 prefetch for the second scratchpad address
immediately after the AES result, before the first address's tweak/store. The
production source was not changed. Both executables used the same portable
reference library and runtime-dispatched AES path. A/B/B/A used three seconds
per CN variant per leg, one worker, fixed input generation and a reference parity
check per variant before timing. No CUDA device was available to this test.

| Variant | Mean candidate change vs control |
|---|---:|
| Dark | -1.1% |
| DarkLite | -13.9% |
| Fast | +14.2% |
| Lite | -3.0% |
| Turtle | -19.6% |
| TurtleLite | -13.3% |

Timing varied substantially within the control (Fast: 91.40 vs 121.25 stage H/s),
so the Fast figure is not a reliable gain estimate. Most variants regress and
there is no full GhostRider, six-worker hybrid, or accepted-work win. Reject this
candidate; do not promote it or spend an overnight run on it.

## Reproducible measurement tool

`cpu-cn-benchmark` is an explicit, excluded-from-default-build target. Build it
inside the existing build-release directory and run it on an idle host:

```bash
cmake --build build-release --target cpu-cn-benchmark --parallel 2
./build-release/cpu-cn-benchmark 3
```

It validates four deterministic inputs per CN variant against the pinned
reference before timing. It never initializes CUDA, connects to a pool or writes
tuning caches. Its stage H/s are not the full miner's H/s. Run baseline/candidate
in A/B/B/A order with the same inputs and compiler options. A passing screen is
only a reason to proceed to six-worker/full-GhostRider testing and then a matched
long hybrid run, not a reason to change production defaults.

## Build resources

Nested NVCC architecture parallelism previously used every CPU for each active
CUDA compilation. 1.0.0.7 defaults `YERBAS_NVCC_THREADS` to 1, with an explicit
non-negative integer override (0 retains all-core behavior). Use build-level
`--parallel 2` on memory-constrained hosts. This does not narrow the generic
PTX/cubin targets, enable native CPU flags, or change generated mining behavior.

## Next gate

Keep blocking waits enabled and batch growth/overlap off for the control.
Before another CPU kernel candidate, establish a low-noise fixed-work baseline
on the mining host, then profile the slow Fast/Lite dependency chains. Earlier
split-multiply, interleave, shuffle-first, table-order, larger-batch and GPU
contention experiments remain rejected. No 1.0.0.7 hash-rate gain is claimed.
