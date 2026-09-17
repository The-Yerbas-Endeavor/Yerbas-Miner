# Yerbas Miner Performance State

Last updated: 2026-09-16 PDT

This file is the canonical resume point for the current Yerbas-Miner optimization campaign. A new chat/session should read this file before proposing new performance work.

## Repository

- Repo: `The-Yerbas-Endeavor/Yerbas-Miner`
- Current experiment branch: `feature/cpu-cn-combo-benchmark`
- Current known branch head before this ledger commit: `017acd2a579da532ba7ff7297f042b29dae326b3`
- Stable overnight branch: `stable/overnight-2026-09-15`
- Stable overnight pinned commit: `cfc57e770aaf433b71bdd1de852c7d9526c79819`

## Production performance goal

- Do not ship tiny wins.
- Target: at least `+100 H/s` sustained whole-miner improvement over comparable long-run baseline.
- Approximate stable long-run whole-miner baseline: `~1.86-1.87 kH/s`.
- Desired sustained range: `~1.97-2.02+ kH/s` on comparable rotations.
- Pool-side accepted-difficulty effective hashrate over exact active intervals is the final authority for production validation.

## Stable overnight evidence

Stable overnight run on the pinned branch ran about 9.5 hours and ended near `1.87 kH/s` average.

- No fatal/CUDA/segfault failures.
- Rejects were stale-job error 21, not invalid hashes.
- This is the stability reference, not necessarily the fastest CPU policy.

## CUDA experiments already tested / do not casually repeat

### 2-lane split multiply — KILLED

- Branch: `feature/cn-2lane-splitmul`
- Commit: `94e1d6457477f41754c255339cbb6973e305f76c`
- GPU0 about `-1.07%` vs baseline.
- GPU1 about `-1.06%` vs baseline.
- Parity passed.

### Dual-state / interleave — KILLED

- Branch: `feature/cn-8lane-interleave`
- Registers/geometry regressed.
- Roughly `-0.8% to -1.0%` in controlled testing.
- Do not resurrect without fundamentally new evidence.

### 4-lane split-multiply — NO PRODUCTION WIN

- Branch: `feature/cn-4lane-splitmul`
- Kernel commit: `9edf426f3fa8aadfe3948137c2582d235780079c`
- Hardened helper head later: `5a5abddb06508e1cf517a10df732c7ac31cb699b`
- Workday run did NOT actually run the splitmul candidate in production; it ran normal `4-lane-ttable` mode.
- Internal selector did not produce a convincing structural win.

### Shuffle-first AES/T-table — KILLED

- Branch: `feature/cn-fast-shuffle-first`
- Commit tested: `b661797b346fe2ac870cb851d89eadcdeb32565f`
- GPU0 about `+2.7%`.
- GPU1 about `0.0%`.
- Parity passed.
- Below structural threshold; no rescue attempt.

### Multi-GPU contention — CLEARED

Three solo-vs-concurrent CN-Fast passes:

- retention `99.55%`
- retention `99.26%`
- retention `100.03%`
- median loss only about `4.29 H/s`

Conclusion: dual-GPU concurrency is not the missing `100+ H/s`.

### 32-bit cooperative multiply note

A cooperative 32-bit multiply direction was already explored historically (Sept. 4 era, commit family including `3153190...`). Do not rebuild the same idea under a new name without first comparing implementation details and old `mul32` selector data.

## GPU tuning conclusions so far

- Ordinary geometry/stagger/cache micro-tuning is largely exhausted.
- CN-Fast geometry `800` vs `768` was only around `0.15%` in clean historical comparison.
- Historical forced CN-Fast often reached roughly `495-505 H/s/GPU`; recent clean forced tests were more like `470-485 H/s/GPU`.
- That gap is worth tracking, but archive analysis has not shown a lost `100 H/s` CUDA regression.
- Later same-CN comparisons generally show GPU performance at least comparable to older builds.

## Pool reconciliation

Sept. 16 workday run:

- Actual mining began about `15:40:44 PDT`.
- The earlier apparent 4.5-hour autotune was actually about 4 hours sitting at an unanswered first-run prompt plus about 0.5 hour of tuning.
- Miner accepted-share events reconcile closely with pool worker CSV after accounting for dev-fee worker `ymdev`.
- Pool was not losing submitted shares.
- Exact active-overlap pool-side effective rate was about `1.667 kH/s` while miner internal final AVG reported about `1.94 kH/s`; this reporting-method discrepancy still needs longer uninterrupted validation.

## Critical CPU finding from restored logs

The restored log archive contained 117 files and thousands of rotation measurements. The strongest unexploited measured lead is CPU policy selection by CryptoNight combination.

### Fresh Sept. 16 CPU tune

The fresh CPU tune found:

- workers: `6`
- batch: `16`
- widths: `1/4/1/2/4/1`
- affinity: `unpinned`
- final validated throughput: `432.70 H/s`

The same tune measured:

- unpinned: `457.16 H/s`
- physical-first: `393.49 H/s`

So the fresh tune strongly preferred `unpinned`.

### Important runtime-learning problem

During the Sept. 16 workday, `YERBAS_CPU_RUNTIME_LEARN` behavior was active and the miner emitted roughly 3,100 CPU fingerprint tuning records while mining. That means real mining time was spent probing deliberately inferior worker counts and width combinations.

Normal production is intentionally supposed to disable this live probing (commit `c24df640a521cb12f380dbef3a0ea3cf53aa7240`, `CPU: keep live fingerprint tuning out of normal mining`).

The historical live-learning data nevertheless showed context-specific CPU width gains large enough to justify a deterministic offline benchmark.

## CPU CN-combination benchmark

Current branch: `feature/cpu-cn-combo-benchmark`

Purpose:

- test all 20 unordered three-CryptoNight combinations;
- compare safe CPU width policies offline;
- parity-check candidates;
- do not initialize CUDA;
- do not connect to the pool;
- do not write a production combination policy.

### First run result — INTERESTING BUT INVALID AS FINAL BASELINE

Uploaded log: `cpu-cn-combo-20260916-222700.log`

First-run benchmark reported:

- combinations tested: `20/20`
- promoted candidates: `12`
- equal-combo baseline: `596.52 H/s`
- equal-combo selected: `711.22 H/s`
- headline projected CPU gain: `+19.23%`
- parity passed for all reported candidates

Large examples included:

- Dark + DarkLite + Turtle: `+46.71%`
- Dark + DarkLite + TurtleLite: `+41.80%`
- DarkLite + Turtle + TurtleLite: `+41.52%`
- Dark + Turtle + TurtleLite: `+35.92%`

However, DO NOT treat `+19.23%` as a production gain yet.

Two flaws were identified:

1. The benchmark accidentally loaded an older CPU cache:
   - widths `1/1/1/1/2/1`
   - affinity `physical-first`
   - throughput `393.09 H/s`

   It therefore rediscovered some gains already captured by the newer `432.70 H/s` policy.

2. The first benchmark summary could still count a candidate after confirmation even when the confirmed candidate was slower than baseline. The rerun must hard-fallback confirmed losers/sub-threshold candidates to baseline.

### Hardened rerun state

Branch was updated to harden the rerun against stale baseline cache and confirmation losers.

Known branch head after that hardening:

`017acd2a579da532ba7ff7297f042b29dae326b3`

The runner is intended to:

- prefer the completed Sept. 16/full-tune cache;
- use the best known CPU baseline instead of stale physical-first cache;
- avoid live learning / fresh retune;
- use more confirmation passes;
- show a confirmation-safe summary that falls back to baseline for losers.

## Immediate next action

Do NOT start another CUDA architecture experiment yet.

First rerun the hardened CPU CN-combination benchmark against the fresh CPU baseline.

From the repo:

```bash
cd ~/Yerbas-Miner
git fetch origin
git switch feature/cpu-cn-combo-benchmark
git reset --hard origin/feature/cpu-cn-combo-benchmark
bash scripts/run-cpu-cn-combo-benchmark.sh
```

Before trusting the rerun, verify startup shows the correct baseline family:

- workers `6`
- batch `16`
- widths `1/4/1/2/4/1`
- affinity `unpinned`
- throughput around `432.70 H/s`

If it instead shows the stale `1/1/1/1/2/1`, `physical-first`, `393.09 H/s` policy, STOP and fix the cache selection before interpreting results.

## Decision rule after hardened CPU rerun

- If incremental combination-aware gain beyond the fresh `432.70 H/s` baseline is trivial (<3%), kill this direction.
- If repeatable incremental CPU gain is `~5-10%`, it is useful but not enough by itself; combine with retained GPU wins.
- If repeatable incremental CPU gain is `>=15%`, prioritize productionizing a cache-only CN-combination CPU policy before more CUDA rewrites.
- Production mining must never do live exploratory probing.

## Working rules

- Whole-miner sustained H/s matters more than isolated forced-benchmark H/s.
- Exact rotation fingerprints / same CN schedules are preferred for comparisons.
- Pool-side accepted-difficulty effective H/s is the final production authority.
- `gpu_tune=auto` for production; `full` only when deliberately retuning.
- Do not lower promotion thresholds to make a candidate look successful.
- Do not spend 20-minute CUDA builds on tiny geometry/stagger ideas.
- Foreground miner runs are preferred for long validation.
- No Docker.
