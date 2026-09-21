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

