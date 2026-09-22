# Production Latency / Hybrid / SASS State

Last updated: 2026-09-17 PDT

This branch follows the completed CPU CN-combination production A/B. The combo
policy is retained as a measured stackable CPU optimization, but it is not the
primary performance breakthrough.

## Branch

- Branch: `feature/production-latency-telemetry`
- Base: `1bf1d587dec33051520e8d7605d513d1b4b6511a`
- Base already records the final CPU combo A/B.
- This branch intentionally starts with measurement-only production telemetry.

## Why production latency is first

The current miner counts a completed GPU batch in raw `hashes_done_` even when
a Stratum job changed before that batch could contribute useful work. Candidate
shares from an obsolete job are correctly suppressed, but the compute time has
already been spent.

The long baseline showed frequent job changes, so raw batch throughput may not
be the same as useful pool work. We must measure the size of this effect before
changing batch policy.

## Telemetry implementation

Opt-in environment variable:

```
YERBAS_PRODUCTION_LATENCY_TELEMETRY=1
```

No scheduler or batch-size policy changes are made when it is enabled.

Per Stratum job:

```
[latency job] new_job=... generation=... prior_job=... prior_lifetime_ms=... clean=...
```

Per completed GPU batch:

```
[latency batch] GPU N job=... generation=... rotation=... CN=...
                hashes=... queue_ms=... scan_ms=... wall_ms=...
                stale=yes|no useful_device_pct=...
```

Periodic status also prints per-GPU cumulative useful/stale hash percentages.

Analyzer:

```
python3 scripts/analyze-production-latency.py logs/production-latency-*.log
```

It reports total useful/stale hashes, stale scan time, job-lifetime distribution,
per-GPU results, per-CN-combination results, and an upper-bound recovery estimate.

## First telemetry run

Use:

```bash
bash scripts/run-production-latency-telemetry.sh
```

The runner:
- performs one CUDA build in `build-production-latency`;
- uses the best qualified cached default CPU policy;
- disables live CPU learning and all retune controls;
- disables the CPU combo policy for a clean baseline;
- enables latency telemetry;
- runs the miner in the foreground;
- logs to `logs/production-latency-YYYYMMDD-HHMMSS.log`;
- automatically runs the analyzer when the miner exits.

The first run does not need another 8-10 hours. About 60-120 minutes should
normally contain enough job transitions to classify whether stale GPU work is a
small or large opportunity.

## Latency decision rule

- stale useful-work loss <3%: kill latency-bounded batching as a major direction;
  continue to hybrid contention and SASS.
- ~3-5%: only test a latency cap if the smaller batches preserve raw GPU
  throughput closely enough to win net useful H/s.
- >=5-8%: prioritize an adaptive latency-bounded batch candidate. This is large
  enough to plausibly contribute materially toward the >=100 H/s project goal.
- Do not change production batch sizes before this measurement.

## Hybrid CPU/GPU contention work

Diagnostic overrides were added after normal CPU autotune, so the qualified CN
width/lane cache remains intact:

```
YERBAS_CPU_WORKERS_OVERRIDE=N
YERBAS_CPU_AFFINITY_OVERRIDE=unpinned|physical-first
YERBAS_GPU_HOST_CPU_MAP=0:CPU,1:CPU
```

The GPU map is device-id to logical-CPU-id. It is opt-in and has no effect on
normal production.

Prepared probes:

```bash
bash scripts/run-hybrid-worker-matrix.sh
bash scripts/run-hybrid-affinity-ab.sh
```

The worker matrix defaults to 4/5/6 CPU workers for 15 minutes each. The affinity
A/B compares the qualified normal worker count against a diagnostic layout that
reserves one detected physical core from CPU mining and pins the two GPU host
scan threads to that core's logical siblings.

Run these after the first latency measurement, not at the same time.

## SASS-first CryptoNight phase-2 work

Historical commit already verified:

`31531906909efb04960d21bc3385f4083d1dbc48`
`CUDA CN: test cooperative 32-bit multiply candidate`

That implementation already distributed four 32x32 partial products across four
lanes and reconstructed the 128-bit product in lane 0. Do not rebuild the same
idea under a new name.

Prepared current-binary inspection:

```bash
bash tools/inspect-cn-phase2-sass.sh
```

It dumps:
- CUDA SASS;
- CUDA resource usage;
- a phase-2 instruction summary focused on multiply/add/shuffle/load/store
  instructions.

Use that result to decide whether a genuinely new phase-2 dependency reduction
exists. Do not write another multiply candidate until the machine instructions
justify it.

## Sept. 20 production / structural results

The regression hunt is now much narrower.

- A long `legacy50` production run held about `1.92 kH/s`, above the historical
  `~1.87 kH/s` floor. Do not describe the current campaign as merely recovering
  old hash.
- `gpu_tune=auto` was incorrectly launching live CN selector/block-size
  tournaments on exact cache misses. Commit `9e2582a` changed auto/off to use an
  exact cache when available and otherwise take the safe four-lane baseline
  immediately. Explicit `gpu_tune=full` remains the retune path.
- Session logging is now timestamped by default (commit `1d5b68a`).
- Historical rev2 stagger policy was restored as diagnostic mode
  `YERBAS_CUDA_OVERLAP=legacyauto` (commit `cc01338`). It did not reveal a
  missing large gain.
- GPU1 Dark/Fast/Lite single-vs-rev2-33/67 A/B:
  single average `604.84 H/s`, legacy33 average `607.71 H/s`; only about
  `+0.47%`, with one stagger pass slower and one faster. Treat as noise-sized;
  stop spending cycles on stagger tuning.
- Raw batch sweep for GPU0 Dark/Fast/Lite proved `3584` is the best tested
  uniform batch:
  `3584=602.85 H/s`, `3712=556.81`, `3840=593.23`,
  `3968=595.19`, `4096=600.40`. Larger batches also increase scan latency.
- Benchmark-only stage-local batching tested outer `7168` with
  Dark/Lite at 7168 and Fast chunked `3584+3584`. Result: `600.07 H/s`,
  about `-0.46%` versus the `602.85 H/s` baseline, with roughly double scan
  latency. KILL this direction.
- Lane-distributed multiply candidate (`mul4`) passed parity and reduced
  registers `32 -> 30` with zero local memory, but lost performance:
  baseline A/B average `597.18 H/s` versus candidate `594.86 H/s`
  (`-0.39%`). Fast phase time regressed about `+0.62%`; Lite about
  `+0.61%`. Do not promote.
- Full overnight rotation analysis shows the four slowest CN triples all contain
  the `CN-Fast + CN-Lite` pair. The current miner matches the old stable build
  on comparable rotations, so this is a structural weak family rather than a
  newly introduced global regression.

Current conclusion: batch policy and stagger policy are no longer the primary
suspects. The remaining high-ceiling work is inside the shared Fast/Lite
CryptoNight phase-2 dependency chain.

### Sept. 20 sm_61 SASS finding

The Pascal SASS dump changes the diagnosis:

- Baseline Fast/Lite four-lane T-table loop: ~309 static instructions,
  55 XMAD-family instructions, 32 registers, zero local memory.
- The parity-correct mul4 candidate reduced this to ~282 static instructions,
  29 XMAD-family instructions and 30 registers, yet ran ~0.4% slower.
- Therefore raw instruction count, multiply instruction count, and register
  pressure are not the primary limiter.
- In the baseline inner loop, both dependent random global loads are consumed
  almost immediately. The second LDG in particular has only a couple of trivial
  instructions before its first dependent shuffle. This points to random-memory
  dependency latency / warp head-of-line blocking as the dominant machine-level
  limitation.
- Exact historical geometry sweeps at batch 3584 already showed block sizes from
  32 through 1024 within only tenths of a percent. Do not reopen geometry tuning.

The split-warp `pairload` diagnostic was tested and KILLED:

- baseline A/B average: `602.13 H/s`
- pairload average: `598.77 H/s` (~`-0.56%`)
- Fast phase regressed about `+0.73%`
- Lite phase regressed about `+1.10%`
- registers increased `32 -> 38`, local memory remained zero

Interpretation: rearranging 4-lane subgroups inside one warp does not provide
independent execution; Pascal still schedules the warp as the unit. Do not build
a second-load version of the same cohort idea.

The two-lane shared-T-table diagnostic also passed parity but was KILLED:

- baseline A/B average: `594.26 H/s`
- two-lane average: `592.40 H/s` (~`-0.31%`)
- Fast phase regressed about `+0.51%`
- Lite phase regressed about `+0.47%`
- registers increased `32 -> 36`, local memory remained zero

Interpretation: doubling hashes per warp did not improve the real loop enough to
offset the extra per-lane AES work and register pressure. Do not promote.

The read-only/L1 T-table diagnostic passed parity but was KILLED:

- baseline A/B average: `603.21 H/s`
- read-only average: `600.48 H/s` (~`-0.45%`)
- Fast phase regressed about `+0.57%`
- Lite phase regressed about `+0.53%`
- registers increased `32 -> 38`; shared memory dropped `4096 -> 0`

Conclusion: the existing 4 KiB shared-memory T-table placement is already better
than Pascal's read-only/L1 path for this workload. Combined with the previous
batch, stagger, geometry, mul4, pairload and two-lane results, stop creating
speculative Fast/Lite phase-2 kernels without new profiler evidence.

### Next target: setup/final launch geometry

The selected register-word setup/final backend still launches at a historical
fixed 128 threads per block. Unlike phase-2, this geometry has not had a
real-batch production sweep.

Commit `323a721` adds diagnostic-only `YERBAS_CN_PHASE_THREADS`.
Accepted values are `32,64,96,128`; default production remains `128`.
The 128-thread upper bound is intentional because the selected word32/shared-key
kernels reserve 16 shared key schedules for 8-lane groups.

Test setup/final geometry before changing their AES or Keccak math. These phases
apply to every CN rotation, so a measured win is globally useful even if smaller
than a phase-2 breakthrough.

Preliminary GPU0 real-pipeline sweep on Dark/Fast/Lite @ batch 3584 found a
repeatable winner:

- 32 threads: `611.72 / 611.55 H/s` -> `611.64 H/s` average
- 64 threads: `602.76 / 602.81 H/s` -> `602.79 H/s` average
- 96 threads: `597.09 / 597.85 H/s` -> `597.47 H/s` average
- 128 threads: `602.86 / 602.89 H/s` -> `602.88 H/s` average

32 threads is about `+1.45%` whole-pipeline versus the current 128-thread
default, with the forward/reverse ordering reproduced. Do not promote globally
yet; first confirm with `YERBAS_DIAGNOSTICS=1` that setup/final timings improve
while phase-2 loop timing stays flat.

The first diagnostic attempt was invalid for phase attribution: the benchmark was
accidentally routed through the retained legacy phase profiler, which ran
`byte-aes` setup/final despite the production cache selecting `word32`. Those
phase numbers must not be used to judge the thread geometry. Commits `f756bab`
and `6b8cce7` restore the benchmark's production dispatcher and instrument the
exact production `word32 -> phase2 -> word32` path directly.

The corrected production phase profile confirms the 32-thread signal is entirely
a setup win:

- setup 32 vs 128:
  - Dark: `35.572 ms` vs `51.725 ms` (~`31.23%` faster)
  - Fast: `138.764 ms` vs `180.399 ms` (~`23.08%` faster)
  - Lite: `69.981 ms` vs `95.034 ms` (~`26.36%` faster)
- phase-2 loop changes by only about `0.04%` or less.
- final at 32 is slightly worse than 128:
  Dark ~`0.18%`, Fast ~`0.49%`, Lite ~`0.91%` slower.
- whole-pipeline diagnostic average remains `602.80 H/s` at 32/32 versus
  `594.27 H/s` at 128/128 (~`+1.44%`).

A clean non-diagnostic A/B/A/B then confirmed the production effect using
setup=32/final=128 versus setup=128/final=128:

- baseline: `585.09 / 585.53 H/s` -> `585.31 H/s` average
- tuned: `592.00 / 601.96 H/s` -> `596.98 H/s` average
- average improvement: `+11.67 H/s`, approximately `+1.99%`

Both tuned passes beat their paired baseline. Together with the earlier
forward/reverse 32-thread sweep (~`+1.45%`), setup geometry is considered a
real production optimization rather than benchmark noise.

Full six-variant rollout at batch 3584 then confirmed the same geometry on both
GTX 1080 Ti devices:

- GPU0: Dark, DarkLite, Fast, Lite, Turtle, TurtleLite -> setup=32, final=128
- GPU1: Dark, DarkLite, Fast, Lite, Turtle, TurtleLite -> setup=32, final=128
- setup reductions versus 128 threads are consistently about 25-30%
- final differences stay below the 1% promotion threshold, so 128 remains selected

The phase-geometry cache is therefore populated for every CN variant on both
test GPUs at the current 3584 production batch. Normal auto mode can now start
without a phase-geometry tournament and reuse the measured values directly.

Therefore setup and final geometry must be selected independently. Commits
`644f32b`, `57450fa`, `cb97afb`, and `5962106` add independent setup/final
thread state, a cached real-batch 32/64/96/128 tuner, safe 128-thread cache-miss
behavior for normal auto mode, and phase-specific reporting. Explicit
`YERBAS_CN_PHASE_RETUNE=1` retunes only this geometry without forcing unrelated
GPU tuning.

## CPU combo result retained

Final same-binary A/B:
- policy ON completed-rotation total ~1757.0 H/s;
- policy OFF ~1729.2 H/s;
- CPU delta ~+22.8 H/s;
- whole-miner delta roughly +20 to +30 H/s.

The offline model predicted ~+23.2 H/s CPU for the actual OFF rotation mix, which
matched production closely. The policy is valid but not the >=100 H/s answer.

## Working rule

Optimize useful accepted-work throughput, not impressive isolated benchmark
numbers. Any future candidate must survive whole-miner validation and ultimately
pool-side accepted-difficulty comparison.

## Sept. 20-21 overnight production validation

The clean post-geometry production run completed essentially 14 hours with
normal cache-first GPU tuning and no profiling/retune tournament.

Final status at 13h 59m 19s:

- cumulative miner AVG: ~1.81 kH/s;
- ending instantaneous total: ~2.15 kH/s;
- CPU instantaneous: ~459.66 H/s;
- GPU0 instantaneous: ~849.89 H/s;
- GPU1 instantaneous: ~838.24 H/s;
- accepted: 9029;
- rejected: 5;
- blocks: 13;
- share acceptance: 99.9%.

The prior long-run production reference is ~1.92 kH/s, so this session finished
roughly 110 H/s / 5.7% below that reference. The last instantaneous 2.15 kH/s
must not be used as the comparison number; the cumulative AVG had already
settled around 1.81-1.82 kH/s by the end of the overnight window.

Stability was good. The observed rejects were stale-job responses at Stratum
job transitions, not invalid hashes or CUDA failures. No CUDA crash/OOM/illegal
access or unintended CPU fallback was observed.

The 3584-batch phase-geometry cache loaded correctly on both GPUs at
setup=32/final=128 with startup benchmarking off. However, live larger batches
for variants such as CN-Lite still use exact-batch selector/geometry state and
can therefore fall back to safe 128-thread setup/final behavior when no cache
exists for that count. Do not interpret the 3584 geometry rollout as a
whole-batch-space rollout.

### Main production bottleneck discovered

Completed-rotation telemetry shows the high-ceiling target is the pair
CN-Fast + CN-Lite, not another setup/final micro-tweak.

Across the overnight run:

- rotations containing both Fast and Lite occupied ~6.90h and averaged
  ~1.449 kH/s;
- Fast without Lite occupied ~4.36h and averaged ~1.901 kH/s;
- Lite without Fast occupied ~1.96h and averaged ~1.914 kH/s;
- rotations containing neither occupied ~0.74h and averaged ~2.950 kH/s.

Therefore roughly half of production time was spent in the weakest class.

The current production policy uses one common nonce count for all 18 GhostRider
stages. When a rotation contains Fast, the per-variant policy can select 3584 as
the minimum preferred batch and therefore pin the whole rotation to that count,
even though other stages/variants can sustain larger batches.

### Next campaign: stage-local batch structure

Do not change the production selector yet.

First use the already-existing diagnostic stage-local path to test a structural
alternative:

- keep the known-safe CN scratchpad chunk at 3584;
- increase the outer GhostRider batch;
- let memory-heavy CN stages execute in bounded chunks inside the existing
  scratchpad budget;
- allow the 15 conventional stages to process the larger outer count.

This is materially different from simply forcing a larger common batch and it
avoids requiring >11 GiB scratchpad allocation for heavy-CN rotations.

Runner added on the active branch:

`scripts/run-fastlite-stage-local-matrix.sh`

First-pass outer candidates:

`3584 4480 5376 6272 7168`

Target triples:

- Dark/Fast/Lite
- DarkLite/Fast/Lite
- Fast/Lite/Turtle
- Fast/Lite/TurtleLite

Both GPUs must be tested. Retuning remains OFF for this first structural matrix.
If stage-local batching produces a repeatable cross-GPU/cross-triple gain, only
then tune the winning outer/chunk combination and consider a production policy.

### Sept. 21 full Fast+Lite stage-local matrix result

The full dual-GPU/four-triple matrix confirms the earlier single-case result and
closes the stage-local batching direction.

Test conditions:
- GPUs: 0 and 1 (GTX 1080 Ti);
- CN chunk: 3584;
- outer sizes: 3584, 4480, 5376, 6272, 7168;
- retuning OFF;
- diagnostics OFF;
- target triples:
  Dark/Fast/Lite,
  DarkLite/Fast/Lite,
  Fast/Lite/Turtle,
  Fast/Lite/TurtleLite.

Across all eight GPU/triple combinations, the ordinary 3584 common batch
remained the winner. Stage-local 3584 reproduced baseline within noise
(average delta about -0.016%). Every larger outer batch was slower on every
GPU/triple combination.

Average delta versus each section's normal 3584 baseline:
- outer 4480: about -0.286%;
- outer 5376: about -0.437%;
- outer 6272: about -0.418%;
- outer 7168: about -0.207%.

The conventional 15 GhostRider stages account for only about 0.033-0.037% of
the measured pipeline time in these Fast+Lite rotations. Roughly 99.96% of the
time remains in the three CryptoNight stages, so increasing the outer count
cannot create a meaningful whole-pipeline gain. Tail CN chunks also introduce
exact-batch selector/geometry state changes at non-3584 counts.

Conclusion: KILL stage-local/outer-batch tuning as a performance direction.
Keep the code path diagnostic-only. Do not productionize it.

### Next GPU action: profiler evidence on Fast/Lite phase-2 stalls

The remaining high-ceiling target is still the shared Fast/Lite phase-2 random
memory dependency chain. Previous SASS work suggests dependent global-load
latency / warp head-of-line blocking, but we now require runtime profiler
evidence before designing another kernel.

Runner added:

`scripts/run-fastlite-phase2-profiler.sh`

It profiles the production
`cryptonight_loop_stage_ttable4_coalesced` kernel at batch 3584 for CN-Fast
and CN-Lite on both GPUs, with all experimental kernels and retune controls off.
The preferred path uses Nsight Compute sections:
- SpeedOfLight;
- SchedulerStats;
- WarpStateStats;
- MemoryWorkloadAnalysis;
- Occupancy.

If Nsight Compute is unavailable, the runner attempts an nvprof analysis-metric
fallback.

Metrics needed before a new kernel experiment:
- long-scoreboard / memory-dependency stalls;
- eligible warps per scheduler;
- achieved vs theoretical occupancy;
- L1/L2 behavior and DRAM traffic;
- issue-slot utilization.

Do not write another mul4/pairload/two-lane/read-only variant until this profiler
pass identifies the dominant runtime stall.

### Sept. 21 profiler compatibility finding

The first Nsight Compute pass failed with ERR_NVGPUCTRPERM. Elevating the
profiler cleared that permission barrier, but the next pass reported that
profiling is not supported on device 0.

The test GPUs are GTX 1080 Ti / compute capability 6.1 (Pascal). Current Nsight
Compute no longer supports Pascal performance-counter profiling. The elevated
attempt also showed that this host's sudo configuration ignores `-E`, so the
forced CN-Fast environment was not preserved and the benchmark fell back to the
normal Dark/DarkLite/Fast schedule.

Commit `cb338716` updates `scripts/run-fastlite-phase2-profiler.sh` to:
- detect compute capability 6.x and bypass Nsight Compute;
- locate `nvprof` from PATH or an installed `/usr/local/cuda-*/bin/nvprof`;
- pass the forced Fast/Lite benchmark environment explicitly through sudo;
- scope profiling to the 13th production four-lane phase-2 launch;
- use kernel replay/analysis metrics on Pascal.

If nvprof is not installed, do not install or downgrade a CUDA toolkit blindly.
First capture the current nvcc version and installed /usr/local CUDA trees so the
least-invasive compatible profiler package can be chosen.

### Sept. 21 Pascal nvprof kernel discovery

Trace-only nvprof discovery succeeded on GPU0 CN-Fast and revealed the exact
production phase-2 symbol used by the old profiler. The kernel is the templated
`cryptonight_loop_stage_ttable4_coalesced<2>` specialization, emitted with a
long internal mangled/static prefix.

Observed launch resources at batch 3584:
- grid: 14 blocks;
- block: 1024 threads;
- registers: 32/thread;
- shared memory: 4 KiB;
- local memory: 0;
- loop duration in trace: about 2.18-2.40 s per CN-Fast launch under nvprof.

The 14x1024 shape is consistent with 3584 hashes x 4 lanes = 14336 CUDA threads.
Historical phase-2 geometry sweeps already covered 32..1024 threads and stayed
within tenths of a percent, so do not reopen block-size tuning solely from this
trace.

Commit `06ffe72` updates the profiler to run a trace-only discovery for each
GPU/variant, extract the exact mangled phase-2 kernel symbol, and then use that
exact symbol for the Pascal metric pass. This replaces the failed substring
selector.

### Sept. 21 nvprof filter failure and unfiltered metric plan

Pascal metric enumeration succeeds and exposes the required counters, but every
attempt to scope `nvprof --metrics` with `--kernels` produced a valid profile
container with zero profiled kernels. Trace-only profiling proves the target
phase-2 kernel does launch normally, so this is treated as an old nvprof
kernel-filter compatibility problem rather than a miner/runtime failure.

Do not spend more cycles on kernel-name matching.

Commits:
- `e96784c`: benchmark-only `YERBAS_BENCH_WARMUP_SCANS` override;
- `53d9834`: Pascal metric collection with no kernel filter.

The Pascal profiler now forces Fast/Lite, sets warmup scans to zero, profiles one
measured 18-stage scan unfiltered, and collects a compact first-pass counter set:
achieved occupancy, eligible warps/cycle, issue-slot utilization, IPC,
memory-dependency/throttle/exec stalls, global-load efficiency, DRAM utilization,
and L2 read hit rate.

This guarantees the three forced CN phase-2 launches are present while keeping
replay time manageable. Parse the phase-2 rows after collection instead of
filtering before collection.

### Sept. 21 Pascal metric profiler closed; true dual-hash ILP candidate opened

The unfiltered nvprof metric pass still produced a valid profile container with
zero profiled kernels. Trace-only nvprof continues to work and shows the real
CUDA launch stream, so the application and kernel launches are healthy. The
hardware-counter path is therefore closed for this host/toolchain.

This matches NVIDIA's current lifecycle boundary:
- R580 is the last driver branch supporting Pascal;
- CUDA 12.x is the last toolkit family supporting Pascal compilation;
- CUDA 13 removes nvprof and the legacy CUPTI Event/Metric APIs.

Do not spend more optimization time trying additional nvprof kernel filters or
Nsight Compute variants on this machine.

The next experiment is a benchmark-only true dual-hash ILP phase-2 kernel in
the existing mode-445 slot. The previous mode-445 implementation was only a
baseline alias, despite retaining dual-hash experiment scaffolding.

New candidate behavior:
- one four-lane subgroup carries two independent CN hashes;
- first dependent random loads for hashes A and B are issued before either is
  consumed;
- second dependent random loads for A and B are likewise issued together;
- both independent state chains remain live so useful arithmetic from one hash
  can overlap memory dependency latency from the other;
- parity uses two hashes;
- normal production remains unchanged unless the explicit dual-hash experiment
  gate and deep retune are enabled.

Cache revisions were bumped because mode 445 changed semantics.

Relevant commits:
- 6dfb1e6: add true dual-hash CryptoNight ILP candidate;
- b4686fd: replace CUDA-local lambdas with conservative device helpers;
- 20f3bf2 / 3bc2cb8 / ecdf9c7: wire mode 445, occupancy and resource queries to
  the new kernel;
- 317e27a / c2b48b0 / efaee8d: invalidate old phase-2/block/production caches;
- 2707e6c: add isolated dual-hash ILP A/B runner.

Run:

`bash scripts/run-dualhash-ilp-ab.sh`

Promotion rule:
- parity PASS on both GPUs for Fast and Lite;
- no local-memory spill;
- then require a repeatable >=2% phase-2 win before any production trial.

### Sept. 21 true dual-hash ILP first A/B result

The first full dual-GPU Fast/Lite A/B produced a mixed but useful result.

All four cases:
- dual-hash parity PASS;
- baseline registers 32/thread;
- dual-hash registers 54/thread;
- baseline local memory 0;
- dual-hash local memory 0.

Median phase-2 deltas versus the existing four-lane baseline:
- GPU0 CN-Fast: -2.316% (dual-hash slower);
- GPU0 CN-Lite: +5.022% (dual-hash faster);
- GPU1 CN-Fast: -0.980% (dual-hash slower);
- GPU1 CN-Lite: -0.842% (dual-hash slower).

GPU0 CN-Lite is not dismissed as simple noise: the three dual-hash samples were
2196.954 / 2218.265 / 2219.682 ms, while the baseline median was 2335.563 ms.
However GPU1 CN-Lite is a stable negative result, so the candidate is not a
universal per-variant promotion.

Important geometry observation:
- batch 3584 dual-hash requires 7168 CUDA threads total;
- occupancy-recommended 576 threads/block produces only 13 blocks;
- GTX 1080 Ti has 28 SMs;
- 256 threads/block produces exactly 28 blocks (one per SM);
- 128 threads/block produces exactly 56 blocks (two per SM).

Because the dual-hash kernel raises registers from 32 to 54, occupancy-based
block selection and device-wide block coverage can disagree. The next test must
therefore sweep dual-hash threads explicitly before deciding whether GPU0/Lite
is a one-card special case or whether the current 576-thread geometry is hiding
a cross-device win.

New controls:
- `YERBAS_CN_DUALHASH_THREADS`: benchmark-only explicit dual-hash block size;
- `YERBAS_CN_DUALHASH_ONLY=1`: skip tile64/prefetch candidates and compare
  baseline directly against dual-hash.

Runner:
`scripts/run-dualhash-lite-thread-matrix.sh`

It tests CN-Lite on both GPUs at:
128, 160, 192, 224, 256, 288, 320, 352, 384, 448, 512, 576 threads.

Do not productionize dual-hash yet.

### Sept. 21 dual-hash thread matrix closes the ILP candidate

The controlled CN-Lite dual-hash thread sweep tested 128, 160, 192, 224, 256,
288, 320, 352, 384, 448, 512 and 576 threads on both GTX 1080 Ti cards at
batch 3584.

The original GPU0/Lite +5.022% result did not reproduce. Representative
controlled deltas:
- GPU0 @128: +0.047%;
- GPU0 @256 (exactly 28 blocks / one block per SM): +0.002%;
- GPU0 @320: +0.105%;
- GPU0 @576: +0.023%;
- GPU1 @128: +0.688%;
- GPU1 @256: +0.175%;
- GPU1 @576: -0.075%.

The apparent GPU0 @448 +3.097% result is rejected because its baseline median
jumped to 2277.444 ms while surrounding GPU0 baseline medians were generally
about 2179 ms. The candidate itself was 2206.911 ms, also slower than the
normal ~2178-2200 ms range. This is a slow-baseline artifact, not a repeatable
candidate win.

Conclusion: true dual-hash ILP is parity-correct and proves that two independent
chains per subgroup can replace roughly half the baseline warp population, but
it does not create additional throughput. Keep it diagnostic only; do not
productionize.

### Next phase-2 hypothesis: sparse warps

The production four-lane kernel maps eight independent hashes into each 32-lane
warp. A random-memory miss from one subgroup can therefore hold the entire warp
at the scoreboard and delay seven unrelated hashes. The retired pairload
experiment could not escape this because scheduling is still warp-granular.

New modes test fewer hashes per independently schedulable warp while preserving
the exact baseline per-hash state machine:
- mode 451: 4 hashes/warp (16 active lanes), 896 warps at batch 3584;
- mode 452: 2 hashes/warp (8 active lanes), 1792 warps at batch 3584.

At the occupancy-recommended 1024 threads/block on the GTX 1080 Ti:
- sparse4 -> 28 blocks, naturally one block per 28 SMs;
- sparse2 -> 56 blocks, naturally two blocks per 28 SMs.

Unlike dual-hash, these candidates do not add a second state chain per active
lane. They isolate warp-level head-of-line blocking by trading inactive lanes
for more independently schedulable warps.

Relevant commits:
- 64b197e: add sparse-warp phase-2 kernels;
- 48caa4e: wire modes 451/452 and parity launch geometry;
- 8f7fd0b: generalize the real-batch selector for sparse-warp experiments;
- 0a3354e: add isolated Fast/Lite sparse-warp A/B runner.

Run:
`bash scripts/run-sparse-warp-fastlite-ab.sh`

Do not productionize sparse modes unless parity passes and a >=2% win repeats.

### Sept. 21 sparse-warps closed; lane-classed shared T-table queued

Sparse-warp A/B tested the unchanged per-hash phase-2 state machine with fewer
hashes sharing each warp.

At batch 3584:
- baseline: 8 hashes/warp;
- sparse4: 4 hashes/warp, 896 warps, 28 blocks at 1024 threads;
- sparse2: 2 hashes/warp, 1792 warps, 56 blocks at 1024 threads.

All sparse candidates passed parity, stayed at 32 registers/thread and used no
local memory. Median phase-2 deltas:
- GPU0 Fast sparse4: -1.146%;
- GPU0 Fast sparse2: -0.647%;
- GPU0 Lite sparse4: +0.107%;
- GPU0 Lite sparse2: -0.910%;
- GPU1 Fast sparse4: +0.274%;
- GPU1 Fast sparse2: +0.285%;
- GPU1 Lite sparse4: +0.252%;
- GPU1 Lite sparse2: +0.083%.

Conclusion: increasing independently schedulable warp count does not materially
improve Fast/Lite. Close the warp head-of-line hypothesis as a production path.

Next experiment: lane-classed shared AES T-tables.

The baseline phase-2 kernel uses one 4 KiB shared T-table set. Pascal shared
memory uses 32 fixed four-byte banks. With random table indices, 32 lanes in a
warp can serialize on bank conflicts even when scratchpad concurrency is already
adequate.

Mode 453 expands the four AES tables to 32 KiB by storing eight duplicate lane
classes per logical table entry:
- physical lookup = table[index][warp_lane & 7];
- the low shared-memory address bits now encode a lane class;
- AES values, scratchpad addresses and CryptoNight state transitions are
  unchanged;
- parity uses eight hashes so every lane class is exercised.

32 KiB remains under Pascal's 48 KiB per-thread-block shared-memory limit.

Relevant commits:
- ea42d98: add lane-classed shared T-table kernel;
- 942f478 / f7059c1: wire and audit mode 453;
- 19369c8: expose mode 453 through the real-batch experimental selector;
- 08a300f: add Fast/Lite dual-GPU bank8 A/B runner.

Run:
`bash scripts/run-bank8-fastlite-ab.sh`

Promotion rule remains parity PASS, no local spill, and a repeatable >=2% phase-2
gain before any production trial.

### Sept. 21 lane-classed shared T-table result; phase-2 microsearch closed

The bank8 shared-table A/B tested the baseline 4 KiB shared AES T-table layout
against a 32 KiB lane-classed eight-replica layout at batch 3584 on both GTX
1080 Ti cards.

All four cases:
- parity PASS;
- baseline registers: 32/thread;
- bank8 registers: 39/thread;
- baseline local memory: 0;
- bank8 local memory: 0;
- occupancy recommendation for bank8: 768 threads.

Median phase-2 deltas:
- GPU0 CN-Fast: -0.368%;
- GPU0 CN-Lite: -0.216%;
- GPU1 CN-Fast: +0.088%;
- GPU1 CN-Lite: +0.097%.

Conclusion: reducing cross-lane shared-memory bank conflicts through a large
lane-classed T-table does not materially improve phase-2 throughput. Do not
promote mode 453.

Combined with the completed negative/near-zero results for phase-2 block
geometry, stage-local batching, distributed multiply, pairload/cohorting,
two-lane/shared-table variants, read-only table access, true dual-hash ILP,
sparse warps, and bank8 shared tables, the current four-lane T-table phase-2
kernel is treated as the best proven Pascal production kernel on this hardware.

No more phase-2 micro-kernel experiments should be scheduled without genuinely
new evidence or profiling capability.

### Best-known-stack production validation

The previous ~14 hour production run averaged about 1.81 kH/s but did not load
the validated CPU CN-combination policy. That policy independently demonstrated
about +20-30 H/s whole-miner improvement.

New runner:
`scripts/run-best-stack-production.sh`

It:
- explicitly disables every benchmark-only CUDA experiment;
- leaves the proven cache-first GPU production policy in place;
- loads `docs/development/cpu-combo-policy-20260916.txt`;
- reuses the strongest known default CPU cache;
- enables production-latency telemetry;
- runs both latency and rotation analyzers after exit.

Recommended next validation window: 6-12 hours.

Relevant commits:
- ee1b0ed: add best-known-stack production validation runner.

### Sept. 21 best-known-stack production snapshot (~69 min)

A live production snapshot with the proven CPU combo policy enabled and all CUDA
experiments disabled ran from about 14:08 to 15:18 local time.

Startup confirmation:
- CPU combo policy loaded: 6 rules, experimental=yes;
- cached CPU policy: 6 workers, batch 16, widths 1/4/1/2/4/1, ~432.70 H/s;
- GPU tuning: auto/cache-first;
- both GTX 1080 Ti cards active with native CUDA GhostRider coverage.

Observed completed-rotation totals across 26 completed rotations (~68.93 min):
- weighted whole-miner throughput: ~1904.8 H/s;
- weighted CPU contribution: ~434.9 H/s;
- weighted GPU contribution: ~1469.9 H/s.

Dashboard cumulative AVG reached ~2.06 kH/s mid-run and was ~1.96 kH/s near
the end of the uploaded snapshot. This is encouraging versus the earlier long
run, but the snapshot is too short and rotation-mix-sensitive to declare a new
sustained production baseline. Keep the current run unchanged for 6-12 hours.

Correctness/stability in the snapshot:
- no CUDA errors, OOM, illegal access, assertion, or CPU fallback;
- zero pool share rejections;
- one block found;
- stale candidates were suppressed locally as designed.

New dominant optimization signal:
- GPU0 stale hashes ~5.1%;
- GPU1 stale hashes ~5.4%;
- stale work is now materially larger than any surviving phase-2 micro-kernel
  delta.

Approximate stale-hash composition in the snapshot:
- 3584-hash scans: about half of stale hashes;
- 5376-hash scans: about one third;
- rare large 11648/17920 scans contribute a disproportionate remainder.

The next engineering target, if the 6-12h run preserves >4% stale device work,
should be job-change-aware batch sizing / stale-cost control rather than another
phase-2 kernel variant.

Commit b82bbc4 extends analyze-production-latency.py to report stale waste by
batch size and by GPU/batch without changing miner behavior.

### Sept. 21 best-stack runner geometry correction

The first ~69-minute best-stack production snapshot confirmed the CPU combo
policy and cache-first GPU path, but it also exposed that setup/final phase
geometry was falling back to 128/128 instead of the previously validated
32/128 geometry at batch 3584.

Representative startup lines showed CN-Fast setup and final as
`phase geometry safe default ... threads=128` on both GPUs.

This means the first snapshot is useful for CPU/stale-work analysis but should
not be treated as the final best-known GPU validation.

Root cause: the best-stack runner selected a cache root primarily to reuse the
strongest CPU policy cache. That cache root did not contain the validated phase
geometry entries, and the runner explicitly disabled phase-geometry retuning.

Commit 4d7aa16 changes only the best-stack validation runner:
- `YERBAS_CN_PHASE_RETUNE=1` is enabled;
- setup/final geometry retunes once per CN variant on first use;
- phase-2 kernel selection remains cache-first;
- all experimental CUDA paths remain disabled.

The next long validation should therefore exercise the actual best-known stack.

### Sept. 21 corrected best-stack snapshot exposes stale-work bottleneck

The corrected best-stack production run beginning 15:26 local time loaded the
validated CPU combo policy and both GTX 1080 Ti cards in cache-first mode.

At about 23 minutes:
- instantaneous whole-miner rate: ~1.99 kH/s;
- cumulative AVG: ~1.91 kH/s;
- GPU0 stale hashes: 12.81%;
- GPU1 stale hashes: 14.05%;
- accepted shares: 331, rejected: 1;
- block found: 1.

The stale loss is now much larger than any surviving CUDA micro-kernel delta.

Observed batch-size stale rates in the uploaded sample:
- batch 3584: 26 stale / 388 completed (~6.7% stale hashes);
- batch 5376: 20 stale / 99 completed (~20.2%);
- batch 11648: 3 stale / 9 completed (~33.3%);
- batch 17920: 3 stale / 7 completed (~42.9%).

The most obvious pathological event was a short ~15 s job while the pure
512-KiB rotation selected 11648 hashes on GPU0 and 17920 on GPU1. The job
changed while both scans were still running, wasting ~9.75 s on GPU0 and
~14.43 s on GPU1.

This does not prove that globally shrinking batches is a win: larger batches
also have higher raw H/s. The correct production metric is useful H/s after
stale work, not stale percentage alone.

### Stale-aware batch crossover experiment

Branch: `feature/stale-aware-batching`

Experimental controls:
- `YERBAS_GPU_STALE_BATCH_CAP`: global production batch cap;
- `YERBAS_GPU_STALE_BATCH_CAP_<device>`: per-device override;
- unset/0 means normal fully adaptive production.

The cap is applied inside `BatchEngine::upload_job()` after the normal
variant/class policy and memory safety limit, but before the batch-keyed
CryptoNight runtime selectors are activated. This preserves selector/cache
semantics at the actual capped count. Benchmark paths ignore the stale cap.

Initial test cap: 5376 hashes. It is already a normal production count on these
cards, aligns exactly to the 896-hash Pascal device quantum, and avoids the
11,648/17,920-hash long scans without forcing Fast rotations below their normal
3584 count.

Runner:
`scripts/run-stale-batch-crossover-ab.sh`

Default design:
- Phase A, 15 min: GPU0 adaptive, GPU1 capped at 5376;
- Phase B, 15 min: GPU0 capped at 5376, GPU1 adaptive;
- same CPU combo policy and cache root;
- all other CUDA experiments disabled;
- production latency telemetry enabled.

The analyzer now reports raw H/s and useful H/s after stale work. A real stale
cap win must follow the capped role when it swaps GPUs and improve useful H/s,
not merely reduce stale percentage.

Relevant commits:
- e7698e9: per-GPU stale-work batch cap;
- 46419cc: repair/extend latency analyzer with useful H/s;
- 6433728: 30-minute crossover production A/B.

### Sept. 21 first fixed-role stale crossover was not informative yet

Phase A (GPU0 adaptive, GPU1 cap=5376) completed about 15 minutes, but the
observed workload was almost entirely Fast-heavy:
- 360 completed GPU batches;
- every completed batch was 3584 hashes;
- GPU0 useful ~718.50 H/s;
- GPU1 useful ~742.19 H/s;
- total stale hashes ~2.22%.

Because the normal adaptive policy was already 3584 for those rotations, the
5376 cap never activated. This phase therefore validates that the cap is
non-invasive below its threshold, but it does not measure the benefit/cost of
capping the 11648/17920 large-batch rotations.

Phase B upload was only a few minutes into its 15-minute window and likewise had
only 3584-hash batches at the time of inspection. Do not infer a cap result from
the partial fixed-role crossover.

### Continuous live crossover replacement

The next experiment removes the restart/job-mix problem.

New runtime API:
- `BatchEngine::set_stale_batch_cap()` allows a cap to change safely between
  drained job uploads.

New env:
- `YERBAS_GPU_STALE_CROSSOVER_CAP=5376`

When enabled, each Stratum generation assigns the cap to one GPU and leaves the
other fully adaptive. The capped role alternates by generation. Because
`upload_gpu_job()` drains active scans before changing policy, this does not
mutate a running CUDA scan.

Only generations where the adaptive peer actually selects a batch above the cap
count as eligible evidence. Fast-heavy 3584/3584 generations are ignored by the
crossover analyzer.

New analyzer:
`scripts/analyze-stale-crossover.py`

It reports:
- capped vs adaptive raw H/s;
- capped vs adaptive useful H/s after stale hashes;
- stale percentage;
- same-GPU capped/adaptive splits;
- batch-size and CN-combination breakdowns.

New runner:
`scripts/run-stale-live-crossover.sh`

Default observation window: 45 minutes. This should collect enough rotation/job
diversity without restarting the miner between roles.

Relevant commits:
- deeade1 / 00cc32b: runtime stale-cap setter;
- 9629e63: alternate cap by live Stratum generation;
- 000874e: crossover analyzer;
- c3ef023: continuous live crossover runner.

### Sept. 21 live stale crossover result: 5376 is promising

The 45-minute continuous crossover finally exercised large-batch rotations.

Overall production-latency summary:
- 848 GPU batches;
- 3,941,504 GPU hashes;
- 350,336 stale hashes (8.89%);
- median job lifetime ~47.05 s;
- p10 job lifetime ~5.10 s;
- p90 job lifetime ~187.51 s.

There were 34 crossover generations and 5 eligible large-batch generations
where the adaptive peer selected more than the 5376 cap.

Eligible-role totals:
- capped: 161,280 hashes, 16.67% stale, 1230.60 raw H/s, 1025.50 useful H/s;
- adaptive: 404,096 hashes, 20.62% stale, 1235.88 raw H/s, 981.03 useful H/s;
- useful-H/s delta: +4.53% for the 5376 cap;
- raw-H/s delta was only about -0.43%.

Same-GPU split is not yet balanced enough for final production promotion:
- GPU0: capped useful 999.38 vs adaptive 1038.52 H/s (-3.77%);
- GPU1: capped useful 1099.40 vs adaptive 966.13 H/s (+13.79%);
- GPU1 only had one eligible generation in the capped role.

Paired eligible generations were directionally favorable in four of five cases:
- generation 2: short ~15 s job; adaptive 17920 scan ended entirely stale while
  capped 5376 completed useful work before the change;
- generation 3: capped useful ~1099.4 vs adaptive ~1038.5 H/s;
- generation 10: capped ~917.5 vs adaptive ~828.8 H/s;
- generation 28: capped ~1050.3 vs adaptive ~1034.2 H/s;
- generation 32: capped ~1081.5 vs adaptive ~1087.2 H/s (essentially flat/slightly negative).

Conclusion: 5376 is promising and has a far stronger signal than the retired
phase-2 micro-kernel experiments, but five eligible generations are not enough
to hard-code it globally yet.

### Large-batch throughput/latency frontier

Before converting the cap into production policy, measure whether large batches
actually buy meaningful raw throughput.

New runner:
`scripts/run-large-batch-latency-frontier.sh`

It benchmarks two observed large-rotation families:
- Dark/DarkLite/Turtle;
- Dark/DarkLite/TurtleLite.

Candidate batches:
3584, 4480, 5376, 6272, 7168, 8960, 10752, 11648, 13440, 16128, 17920.

Method:
- both GPUs;
- full 18-stage raw common batch;
- setup/final geometry fixed at proven 32/128 for a fair comparison;
- no stale caps or experimental kernels;
- two passes, ascending then descending candidate order;
- analyzer reports median raw H/s, median scan latency, and the smallest batch
  within 99% of peak throughput.

If 5376 (or another small batch) is inside the 99% raw-throughput plateau on
both GPUs/triples, the production autotuner should be changed from
"absolute maximum H/s wins" to "smallest batch inside the throughput plateau."
That generalizes stale protection to other GPUs without hard-coding a Pascal
batch size.

Relevant commits:
- 866bb93: paired-generation crossover reporting;
- 183301c: throughput/latency frontier analyzer;
- ad27164: two-pass large-batch frontier runner.

### Sept. 21 large-batch frontier harness limitation

The first `run-large-batch-latency-frontier.sh` run did not produce a complete
3584..17920 frontier.

The benchmark constructs a fresh raw-batch `BatchEngine` for each requested
size. In raw-batch mode its scratchpad budget is based on
`requested_size * max_scratchpad_stride` (2 MiB/hash), even though the forced
Dark/DarkLite/Turtle and Dark/DarkLite/TurtleLite triples need only 512 KiB/hash
at runtime. As a result, GPU0 could measure only 3584 and 4480 before allocation
failure, and GPU1 could measure only 3584/4480 plus one 5376 sample.

Therefore the generated frontier recommendation of 3584 is not a complete
3584..17920 comparison and must not be treated as such.

The valid samples are nevertheless useful:
- GPU0 Dark/DarkLite/Turtle:
  - 3584: ~1231.88 H/s median;
  - 4480: ~1228.35 H/s;
- GPU0 Dark/DarkLite/TurtleLite:
  - 3584: ~1231.43 H/s;
  - 4480: ~1228.95 H/s;
- GPU1 Dark/DarkLite/Turtle:
  - 3584: ~1257.41 H/s;
  - 4480: ~1255.98 H/s;
  - 5376: ~1256.03 H/s;
- GPU1 Dark/DarkLite/TurtleLite:
  - 3584: ~1257.14 H/s;
  - 4480: ~1253.77 H/s;
  - 5376: ~1255.77 H/s.

Within the sizes that actually ran, larger batches bought no raw-throughput
gain. This is consistent with the live crossover, where 5376 lost only ~0.43%
raw H/s versus 11648/17920 while improving useful H/s by ~4.53%.

### Thresholded 3584-vs-large live crossover

Rather than rely on the incomplete synthetic frontier, the next production test
directly asks whether 3584 is better than 5376 specifically on the large
11648/17920 rotations.

New runtime threshold:
- `YERBAS_GPU_STALE_CROSSOVER_MIN_TUNED`

The experimental cap applies only when the normal tuned batch is at or above
this threshold. This lets a 3584 cap target only genuinely large rotations while
leaving normal 5376 CN-Lite rotations unchanged.

Next test:
- cap: 3584;
- minimum tuned batch: 6272;
- role switches every Stratum generation;
- eligible evidence requires the adaptive peer to use >=6272.

Run:
`YERBAS_STALE_LIVE_CAP=3584 YERBAS_STALE_LIVE_MIN_TUNED=6272 bash scripts/run-stale-live-crossover.sh`

The paired-generation analyzer now includes the threshold in its output.

Relevant commits:
- ba68fc3 / f12bec3 / 1666f21: thresholded runtime cap;
- edb5e50: thresholded per-generation crossover wiring;
- 7e85b7e: threshold-aware crossover analyzer;
- 0013c68: runner support.

### Sept. 21 thresholded 3584-vs-large crossover: strong win

The 45-minute live crossover tested a 3584 cap only when the normal tuned batch
was >=6272, leaving normal 3584 and 5376 rotations untouched.

Run summary:
- 55 crossover generations;
- 6 eligible large-batch generations;
- capped role: 68,096 hashes, 31.58% stale, 1405.33 raw H/s,
  961.54 useful H/s;
- adaptive role: 181,888 hashes, 45.32% stale, 1371.42 raw H/s,
  749.89 useful H/s;
- useful-H/s delta: +28.22% for the thresholded 3584 cap.

Eligible paired generations:
- gen 5: +38.89% useful H/s;
- gen 12: +30.57%;
- gen 37: capped side completed useful work; adaptive side was 100% stale;
- gen 39: +10.99%;
- gen 47: +16.10%;
- gen 48: both sides 100% stale on an approximately 5 s job.

The role balance was not symmetric enough for a same-GPU final verdict:
- GPU0 capped/adaptive useful delta: -8.63% (only 4 capped batches);
- GPU1 capped/adaptive useful delta: +95.46% (only 3 adaptive batches).
However, the same-job paired comparisons are overwhelmingly favorable and the
gen 12 result favors the capped GPU even though GPU0 is the slower of the two
cards.

Whole-run production telemetry still shows substantial stale opportunity:
- 1,015 GPU batches;
- 12.30% stale hashes overall;
- median job lifetime 23.27 s;
- p10 job lifetime 5.09 s;
- theoretical upper-bound uplift if every stale hash were recoverable: 14.02%.

The thresholded cap targets the pathological large scans without touching the
normal 5376 CN-Lite path.

### Corrected large-batch frontier harness

The first frontier harness over-allocated raw-batch scratchpad memory using the
global maximum 2 MiB/hash stride, so 512-KiB rotations could not reach their
real production batch sizes.

Benchmark-only control added:
- `YERBAS_CUDA_BENCH_SCRATCHPAD_STRIDE`

For the Dark/DarkLite/Turtle and Dark/DarkLite/TurtleLite frontiers the runner
now sets the stride to 524288 bytes/hash, matching the maximum active CN
scratchpad size in those forced triples. Production allocation behavior is
unchanged.

This should allow the synthetic frontier to measure 3584 through 17920 using the
same effective scratchpad budget production already exercises.

Relevant commits:
- 2ba2818: realistic raw-benchmark scratchpad budget;
- 5e65fad: corrected frontier runner.

### Sept. 21 stale-aware production candidate

The thresholded 3584 live crossover is strong enough to promote into a
hardware-specific production candidate for the measured GTX 1080 Ti platform.

Default policy on NVIDIA GeForce GTX 1080 Ti (CC 6.1):
- normal tuned batch <6272: unchanged;
- normal tuned batch >=6272: cap to 3584;
- normal 3584 Fast rotations: unchanged;
- normal 5376 Lite rotations: unchanged.

The policy is deliberately restricted to GTX 1080 Ti until other GPU families
are measured. Other GPUs keep the existing adaptive batch policy.

Overrides:
- `YERBAS_GPU_STALE_BATCH_CAP=<n>` explicitly sets the cap;
- setting `YERBAS_GPU_STALE_BATCH_CAP=0` disables the validated default;
- `YERBAS_GPU_STALE_BATCH_MIN_TUNED=<n>` overrides the threshold.

The 3584 setup/final geometry cache-miss path is also specialized only for the
validated GTX 1080 Ti case:
- setup: 32 threads;
- final: 128 threads.
This prevents newly capped Dark/DarkLite/Turtle/TurtleLite rotations from
falling back to the conservative 128-thread setup geometry when no cache entry
exists.

Production-validation runner:
`scripts/run-stale-aware-production.sh`

It clears all benchmark/crossover/stale override environment variables and runs
the default candidate with telemetry enabled. Default observation window: 1 h.

Promotion gate to main:
- automatic policy detected on both GTX 1080 Ti cards;
- large tuned rotations cap to 3584;
- normal 3584/5376 rotations remain unchanged;
- no CUDA errors or CPU fallback;
- no unexpected share rejection increase;
- whole-run useful GPU throughput/stale loss improves or remains clearly better
  than the uncapped best-stack observations.

Relevant commits:
- 919f261: promote validated 1080 Ti stale-aware batch policy;
- e4b9f01: validated GTX 1080 Ti 3584 setup/final geometry;
- f4381ff: production validation runner.

### Sept. 21 production-candidate snapshot: healthy, but threshold tightened

The first shipped-policy production snapshot started at 20:44 and the uploaded
log currently covers about 23 minutes.

Validated:
- automatic GTX 1080 Ti stale-aware policy detected on both GPUs;
- CPU combo policy loaded from cache;
- native CUDA coverage 15/15 cores and 6/6 CryptoNight variants;
- no CUDA errors, OOM, illegal access, assertion, or CPU fallback;
- no rejected shares observed in the uploaded snapshot;
- last complete status block: 406 accepted / 0 rejected;
- GPU stale rate at ~20m: ~2.23% GPU0 and ~2.22% GPU1.

Important limitation:
- every observed rotation in this snapshot contained CN-Fast, so normal
  variant-min batching already selected 3584;
- zero stale-aware cap activations occurred;
- therefore this snapshot validates non-interference/stability when the policy is
  idle, but is not yet the final both-GPUs-large-rotation validation.

New edge case discovered from startup cache state:
- GPU0 CN-Lite cache is currently 7168;
- GPU1 CN-Lite cache is 5376.
The previous automatic threshold of 6272 could therefore cap GPU0's 7168 Lite
path even though the live crossover win was measured on 11648/17920-class
rotations.

The hardware-specific default threshold is tightened to 8960:
- tuned 3584/5376/6272/7168: unchanged;
- tuned >=8960: cap to 3584.
This keeps the production rule aligned with the measured pathological
11648/17920 scan class.

Relevant commits:
- fc945da: narrow 1080 Ti stale cap to large rotations;
- 6d71bf4: update production validation expectation to >=8960.

### Sept. 21 narrowed-threshold live snapshot (~2 min)

The first production candidate with the narrowed automatic threshold
(`cap=3584, minimum-tuned=8960`) started cleanly on both GTX 1080 Ti cards.

Observed behavior:
- CPU combo policy loaded from cache at ~432.70 H/s;
- both GPUs detected the validated 1080 Ti stale-aware policy;
- first Fast/Lite/TurtleLite rotation stayed at 3584 on both cards;
- next DarkLite/Lite/Turtle rotation selected 7168 on GPU0 and 5376 on GPU1;
- no stale-aware cap activation occurred, which is correct because both tuned
  counts are below 8960;
- 7168 and 5376 scans produced accepted shares;
- no CUDA errors, OOM, illegal access, assertion, CPU fallback, or rejected
  shares appeared in the uploaded snapshot.

This confirms the 8960 threshold preserves the mid-size 5376/7168 paths and
targets only the measured long-scan class.

New optimization opportunity observed:
- cache misses at batch 5376/7168 use conservative 128-thread setup geometry for
  some CN variants.
- Do not hard-code 32 threads at these counts yet: the 32/128 validation was
  performed at batch 3584. Measure 5376/7168 independently before changing the
  production fallback.

The final merge gate is still an actual production job where both GPUs'
uncapped variant-min choice is >=8960 and the automatic policy logs
`stale-aware batch cap ... -> capped=3584`.

### Sept. 21 one-hour narrowed-threshold production validation

The ~1 h production candidate with `cap=3584, minimum-tuned=8960`
successfully exercised the automatic stale-aware policy on both GTX 1080 Ti
cards.

Observed automatic cap activations:
- GPU0: 7 events, tuned 11648 -> 3584;
- GPU1: 7 events, tuned 17920 -> 3584;
- 14 cap events total.

The capped large rotations completed near the expected low-latency range
(~2.3-3.0 s scans in the observed 512-KiB rotations) instead of the prior
~9-15 s long scans.

One-hour production-latency analysis:
- GPU batches: 1297;
- completed GPU hashes: 5,173,504;
- useful hashes: 4,644,864;
- stale hashes: 528,640 (10.22%);
- GPU scan time: ~6844.1 s;
- stale scan time: ~674.6 s (9.86%);
- job lifetime median: ~29.99 s, p10 ~8.26 s, p90 ~118.03 s;
- GPU0 raw/useful: ~746.68 / 666.90 H/s;
- GPU1 raw/useful: ~765.59 / 691.02 H/s.

Final visible production status was approximately:
- AVG 1.77 kH/s;
- 505 accepted / 1 rejected at 58m45s;
- 7 blocks at that status, with an eighth block found shortly afterward.

The remaining stale work is now concentrated in ordinary 3584 plus the
preserved 5376/7168 mid-size paths, so the narrowed policy is behaving as
designed rather than indiscriminately shrinking every rotation.

### Overnight endurance validation

The next production-candidate run should cover a full night rather than another
short sample.

`scripts/run-stale-aware-production.sh` now defaults to:
- 36,000 seconds;
- approximately 10 hours;
- production latency telemetry enabled;
- shipped/default stale-aware policy;
- all experimental CUDA controls disabled.

The duration remains overridable with
`YERBAS_STALE_PRODUCTION_SECONDS=<seconds>`.

Overnight validation goals:
- repeated 11648/17920 -> 3584 cap activations across many rotation/job mixes;
- stable accepted/rejected share quality;
- no CUDA errors, OOM, illegal access, assertion, or CPU fallback;
- observe multiple developer-fee windows;
- capture enough job-lifetime distribution to judge remaining stale loss;
- compare sustained whole-run and useful-GPU throughput to the one-hour run.

Relevant commit:
- 959ee42: make stale-aware production validation a 10-hour overnight run by default.

### Sept. 21-22 overnight stale-aware production validation: PASS

The overnight production candidate began at 22:41:38 and the uploaded snapshot
covers approximately 9 h 50 min of uninterrupted mining, effectively the full
10-hour endurance window.

Final visible production status near 9 h 50 min:
- instantaneous total: ~1.91 kH/s;
- cumulative AVG: ~1.92 kH/s;
- accepted: 7,090;
- rejected: 10;
- share acceptance display: 99.9%;
- blocks found: 9;
- GPU0 stale hashes: 5.87%;
- GPU1 stale hashes: 5.43%.

All 10 rejected shares were pool error 21, "stale job". No invalid-hash or
correctness rejection was observed.

Automatic stale-aware policy coverage:
- GPU0 11648 -> 3584 cap activations: 41;
- GPU1 17920 -> 3584 cap activations: 41;
- total cap activations: 82 across 41 large-rotation jobs.

The narrowed 8960 threshold also preserved the intended mid-size paths:
- GPU0 7168 remained 7168;
- GPU1 5376 remained 5376.

The capped 512-KiB large rotations consistently ran near roughly 2.3-3.0 s per
scan instead of the old ~9-15 s large scans.

Overnight telemetry calculated from the uploaded log:
- completed GPU batches: 13,576;
- completed GPU hashes: 53,548,544;
- stale hashes: 3,021,312 (~5.64%);
- job-lifetime samples: 337;
- median job lifetime: ~47.73 s;
- p10: ~5.35 s;
- p90: ~203.02 s;
- GPU0 raw/useful: ~789.80 / 743.55 H/s;
- GPU1 raw/useful: ~791.91 / 748.99 H/s.

For comparison, the prior one-hour production sample had ~10.22% stale hashes.
The rotation/job mix differs, so this is not a strict A/B percentage claim, but
the overnight endurance result confirms that the stale-aware policy remains
stable under a much broader production workload.

The run crossed 10 developer-fee windows and returned to the configured pool
after each completed window. No CUDA error, OOM, illegal access, assertion,
segmentation fault, or CPU fallback was observed.

Production gate result: PASS.

The validated GTX 1080 Ti default is ready for main:
- tuned batch <8960: preserve normal adaptive choice;
- tuned batch >=8960: cap to 3584;
- validated batch-3584 phase geometry: setup=32, final=128;
- explicit environment stale-cap settings remain available as overrides.

Next step: fast-forward main to the validated stale-aware branch and run the
full GitHub Actions build matrix.

### Sept. 22 next target: 5376/7168 phase geometry

After the overnight stale-aware policy passed production validation, the next
safe optimization target is the preserved mid-size 1-MiB paths:
- GPU0 commonly selects 7168;
- GPU1 commonly selects 5376.

Overnight logs show some CN setup/final geometry cache misses at those batch
sizes falling back to the conservative 128-thread launch geometry. The 32/128
specialization validated for batch 3584 must not be assumed valid at 5376 or
7168 without measurement.

Branch:
`feature/midsize-phase-geometry`

New benchmark-only protection:
- `YERBAS_CN_PHASE_NOSAVE=1`
- when enabled, CN phase backend/geometry benchmark results are not written to
  persistent cache.
- production behavior is unchanged when the variable is unset.

Runner:
`scripts/run-midsize-phase-geometry.sh`

Coverage:
- GPUs 0 and 1;
- exact batches 5376 and 7168;
- representative observed triples:
  - Dark/Lite/TurtleLite;
  - DarkLite/Lite/Turtle;
- two independent process repeats per case;
- setup/final geometry candidates 32/64/96/128;
- raw-batch scratchpad budget fixed at 1 MiB/hash to match these triples;
- production stale policy and all unrelated tuning controls are disabled for
  the benchmark.

Analyzer:
`scripts/analyze-midsize-phase-geometry.py`

Promotion gate:
- candidate must satisfy the existing >=1% improvement threshold versus 128;
- result should repeat across both independent runs;
- direction should be consistent across both GTX 1080 Ti cards for the same
  batch/variant/phase;
- do not generalize a 5376 result to 7168 or vice versa without evidence.

Relevant commits:
- c95bc5e: no-save CN phase benchmark mode;
- 0a005c6: mid-size phase geometry runner;
- b876ce8: mid-size geometry analyzer.

### Sept. 22 mid-size phase-geometry result

The no-save 5376/7168 geometry sweep completed on both GTX 1080 Ti cards.

5376 result:
- no cross-card production promotion;
- GPU1 consistently preferred the conservative 128-thread setup/final geometry;
- GPU0 showed a few isolated 96/64 candidates, but several did not repeat across
  runs or did not agree with GPU1;
- keep 5376 production fallback unchanged at 128.

7168 result:
- setup=32 was selected on both GPUs, both repeats, for every tested non-Fast
  CN variant:
  - CN-Dark;
  - CN-DarkLite;
  - CN-Lite;
  - CN-Turtle;
  - CN-TurtleLite.
- setup gains versus 128 were approximately 3.3% to 7.0% at the phase level;
- final geometry remained 128 across the repeatable cross-card result set.

Representative aggregate setup gains at batch 7168:
- GPU0:
  - Dark +5.23%;
  - DarkLite +6.99%;
  - Lite +6.45%;
  - Turtle +4.14%;
  - TurtleLite +3.39%;
- GPU1:
  - Dark +5.94%;
  - DarkLite +6.27%;
  - Lite +5.84%;
  - Turtle +3.26%;
  - TurtleLite +3.34%.

Although the phase-local signal is strong, setup is only a small fraction of a
full 6-8 second GhostRider scan. Do not promote the fallback on phase timing
alone.

Next gate:
`scripts/run-7168-setup-ab.sh`

Whole-pipeline A/B design:
- exact batch 7168;
- both GTX 1080 Ti cards;
- representative triples Dark/Lite/TurtleLite and DarkLite/Lite/Turtle;
- setup=32 versus setup=128;
- final fixed at 128;
- four paired repeats per GPU/triple;
- A/B execution order alternates each repeat to reduce thermal/order bias;
- no persistent cache writes;
- production stale-aware policy unchanged.

Analyzer:
`scripts/analyze-7168-setup-ab.py`

Promotion requirement:
- positive median whole-pipeline H/s delta;
- directionally positive paired results on both cards/triples;
- live-production validation after synthetic A/B before merging.

Relevant commits:
- b14d4ff: batch-7168 setup geometry whole-pipeline A/B runner;
- 237def1: batch-7168 A/B analyzer.

### Sept. 22 batch-7168 whole-pipeline setup A/B: PASS candidate

The full-pipeline A/B compared setup=32 versus setup=128 at exact batch 7168,
with final fixed at 128. Coverage was both GTX 1080 Ti cards, two representative
mid-size rotation families, four paired repeats, and alternating execution
order.

Summary:
- GPU0 Dark/Lite/TurtleLite:
  - median setup32: 871.44 H/s;
  - median setup128: 837.95 H/s;
  - nominal delta +3.997%, but dominated by the first two warm/order-sensitive
    pairs; the last two paired deltas were +0.262% and -0.216%.
  - Treat the +4% headline as non-production evidence.
- GPU0 DarkLite/Lite/Turtle:
  - 890.79 vs 888.98 H/s;
  - +0.204%;
  - all four paired repeats positive (+0.253/+0.209/+0.178/+0.175%).
- GPU1 Dark/Lite/TurtleLite:
  - 896.00 vs 894.08 H/s;
  - +0.214%;
  - all four paired repeats positive (+0.209/+0.167/+0.213/+0.283%).
- GPU1 DarkLite/Lite/Turtle:
  - 895.85 vs 894.30 H/s;
  - +0.173%;
  - all four paired repeats positive (+0.124/+0.190/+0.248/+0.164%).

Overall sample median:
- setup32: 893.31 H/s;
- setup128: 891.36 H/s;
- +0.219%.

Interpretation:
- the credible whole-pipeline benefit is approximately +0.2%, not +4%;
- the direction is repeatable on both cards and both representative rotation
  families after excluding the obvious early GPU0 warm/order transient;
- batch 5376 remains unchanged because its geometry sweep did not meet the
  cross-card consistency gate.

Feature-branch production candidate:
- GTX 1080 Ti + batch 3584: setup=32, final=128 (existing validated policy);
- GTX 1080 Ti + batch 7168: setup=32, final=128;
- batch 5376: conservative/default behavior unchanged;
- other GPU families: unchanged.

Live validation runner:
`scripts/run-7168-live-validation.sh`

Default live window: 2 hours. The runner clears all CN geometry and stale-policy
overrides, enables production latency telemetry, and exercises the branch's
default policy. Live gate:
- observe `source=validated-1080ti-7168` on actual batch-7168 production work;
- no CUDA errors, fallback, correctness rejects, or abnormal stale/share loss;
- 5376 remains unchanged;
- stale-aware >=8960 -> 3584 behavior remains intact;
- sustained production H/s remains at least neutral versus the overnight
  baseline, with the expected improvement small (~0.2% on affected rotations).

Relevant commits:
- fab2b01: promote validated GTX 1080 Ti batch-7168 setup geometry;
- 088be0a: add 2-hour live validation runner.

