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

## Production GPU profile — 2026-10-15 19:16

The bounded 1.0.0.6 control run captured all six CN variants on both GPUs:
292 accepted shares, zero rejected, no CUDA errors, clean shutdown. Across
588.98 seconds of completed rotations, the duration-weighted rate was
1971.22 H/s total (1481.74 GPU, 489.48 CPU). This short instrumented workload,
including a developer-fee pool switch, is not comparable to the overnight mean.

Summed sampled GPU stage times put 99.965% inside CryptoNight. At batch 3584,
Fast took 2524.189/2465.829 ms on GPUs 0/1 (86.74/87.01% loop), and Lite took
2376.516/2306.807 ms (92.89/92.98% loop). Setup/final are secondary; accelerating
the other 15 hashes cannot deliver a useful overall gain on this profile.

## Native-width CPU multiply candidate

Disassembly of the portable CPU AES hot loop showed the pinned `mul128` helper
expanding into four smaller multiplies plus carry handling. The CPU reuse paths
now use an exact unsigned native-wide product when available: `__uint128_t` on
supporting GCC/Clang targets, `_umul128` on MSVC x64, and a portable limb fallback
elsewhere. The existing AES runtime dispatch is preserved; no native CPU build
flags or CUDA architecture changes are required. The pinned reference is
unchanged. All custom 1/2/4-way CN paths use the helper. Startup identifies the
multiply backend separately from AES features.

Correctness gates include both native and fallback products against the pinned
helper for 100 carry/overflow edge pairs and one million deterministic pairs,
plus optimized full GhostRider versus the legacy/reference 96-header corpus.
The existing three CTests pass. Explicit 2-way and 4-way full-plan runs also
pass grouped parity and worker coverage across all 20 combinations.
Full-plan measurement also checks grouped
reference parity across all 20 CN combinations and worker nonce coverage.

Exploratory timing: the serial 3-second-per-stage A/B/B/A showed Fast +4.13%,
Lite +6.87%, Turtle +5.71%, TurtleLite +4.57%; Dark/DarkLite were noisy
(+1.79%/-0.15%). The first six-worker 1-second-per-header A/B/B/A equal-rotation
means were control 515.99/578.97 and candidate 598.61/604.59 H/s (+9.89% mean).
These are CPU-only synthetic scores, not a claimed hybrid/pool improvement.
An earlier overlapping pair of test jobs was excluded from performance evidence.
The compact AES-round and byte-extraction screens were not retained.

The longer repository-build A/B/B/A (six workers, width 1, three seconds per
header, same 192-nonce batches) gave controls 492.46/515.47 and candidates
578.94/581.44 H/s. Their equal-rotation means are 503.965 versus 580.190 H/s:
+15.125%. All 20 combinations have positive paired mean changes, from +1.63%
to +46.60%, though individual rows remain noisy and this is not a statistical
confidence interval. Together with the shorter +9.89% screen, this qualifies the
candidate for live hybrid testing, not a 15% whole-miner claim. No >=100 H/s
sustained whole-miner or accepted-work improvement has been established.

Next gate: rebuild the generic 1.0.0.7 miner and run a bounded uninstrumented
hybrid comparison with the same cached six-worker policy, blocking CUDA waits,
growth/overlap/latency controller off, and GPU tuning auto. Verify startup shows
`mul128=native-wide` (or the appropriate compiler backend), accepted shares from
both GPUs and CPU, and no invalid shares. Preserve tuning caches. Compare
matched rotations and ultimately overnight accepted-difficulty work before
promoting a release. GPU kernels and the 1.0.0.6 release remain unchanged.
