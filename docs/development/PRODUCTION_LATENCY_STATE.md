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

