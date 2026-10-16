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

## Follow-up CPU screens

The mining host's three 10-second single-worker passes established medians:
Dark 382.02, DarkLite 567.28, Fast 146.90, Lite 165.48, Turtle 1136.06 and
TurtleLite 1177.41 stage H/s. Max-minus-min relative to the median was 1.16–2.21%.
These are stage rates, not full GhostRider mining rates.

A temporary scalar-accumulator Fast/Lite kernel passed the initial parity screen
but showed only roughly 1–2% in the short local A/B/B/A. It was not promoted;
production kernel source was restored. No accumulator environment switch ships.

Inspection found the autotuner chooses workers on scalar execution and then
qualifies wider CN paths at that fixed worker count. To test that search gap,
`cpu-plan-benchmark` measures the actual production WorkerPool across a stable
corpus containing all 20 unordered CN combinations. It checks grouped complete
GhostRider hashes against the reference before timing and verifies that worker
nonce coverage matches the reported hash count. Measurement mode disables live
learning/profile cache writes. No pool or CUDA initialization is performed.

The nonce-coverage probe uses a separate header/job key because production pins
the target to that key. Timed runs use an impossible target. Six workers with
batch 32 and three workers with batch 64 both cover the same 192 nonces per batch.

Local diagnostic outcomes (synthetic equal-rotation mean):

- 6 workers × 1-way A/B/B/A control: 569.58 / 560.51 H/s.
- 3 workers × 2-way candidate: 447.92 / 427.52 H/s. Reject (~22.5% lower).
- 6 workers × uniform 2-way screen: 594.46 H/s, but every Fast-containing row
  regressed (roughly 16–33% against the prior controls). Reject the global policy.
- Fresh A/B/B/A, six workers, mixed widths `2,2,1,1,2,2`: control 541.95 / 516.57,
  mixed 596.71 / 616.84 H/s. The equal-combination score improves ~14.6%, but nine
  of ten Fast-containing combinations are lower. Reject a global default change.

The mixed-policy wins concentrate in easy non-Fast combinations, reproducing the
historical combo-policy pattern rather than establishing a new whole-miner
breakthrough. An equal-combination average is not production-time weighting.
One header per combination also does not establish performance for every core
schedule/order. No production cache, tuner threshold or default width changes.

To reproduce a full-plan screen with the miner stopped:

```bash
cmake -S . -B build-release -DYERBAS_NATIVE_CPU=OFF
cmake --build build-release --target cpu-plan-benchmark --parallel 2
for policy in 1 2,2,1,1,2,2 2,2,1,1,2,2 1; do
  env -u YERBAS_CPU_COMBO_POLICY_FILE ./build-release/cpu-plan-benchmark 6 "$policy" 1
done
```

Widths are in Dark/DarkLite/Fast/Lite/Turtle/TurtleLite order. A single width
applies to all six. The tool accepts widths 1/2/4; each chosen grouped path must
pass reference parity before its measurements run. It is excluded from ordinary
builds. Existing caches are preserved.

Next action: obtain a bounded production GPU phase profile using the existing
`YERBAS_CUDA_PROFILE=1` switch with blocking waits, growth/overlap disabled and
cache-first GPU tuning. The diagnostic build must not be treated as a throughput
comparison because event instrumentation adds overhead. No additional CPU
baseline repetition or speculative GPU kernel promotion is justified yet.
