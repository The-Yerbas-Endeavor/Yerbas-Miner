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
