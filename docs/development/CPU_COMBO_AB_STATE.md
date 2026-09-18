# CPU CN-Combination Production A/B State

Last updated: 2026-09-17 PDT

This file records the live production A/B so it can be resumed across chats without relying on conversation memory.

## Policy-ON run

Log: `cpu-combo-production-ab-20260917-001052.log`

Startup was correct:
- CPU cache: workers=6, batch=16, widths=1/4/1/2/4/1, affinity=unpinned, 432.70 H/s
- CPU combo policy loaded with 6 rules
- GPU1 started from full cache
- GPU0 initially started from runtime-safe/class-fallback and populated more cache during the run

At about 8h30m:
- miner AVG: ~1.85 kH/s
- accepted/rejected/blocks: 4874 / 2 / 4
- exact completed-rotation weighted telemetry over ~8.40h: ~1757.0 H/s total, ~371.1 H/s CPU, ~1385.9 H/s GPU

## Policy-OFF baseline run

Final log: `cpu-combo-production-baseline-20260917-092041(1).log`

Same already-built binary, combo policy OFF.

Startup was clean and comparable:
- workers=6
- batch=16
- widths=1/4/1/2/4/1
- affinity=unpinned
- cached throughput=432.70 H/s
- both GPUs started from populated calibration/variant caches

At about 10h19m:
- miner AVG: ~1.81 kH/s
- accepted/rejected/blocks: 7762 / 3 / 11 at the last full status block
- exact completed-rotation weighted telemetry over ~9.92h: ~1729.2 H/s total, ~348.3 H/s CPU, ~1380.9 H/s GPU

## A/B result

Whole completed-rotation telemetry:
- policy ON total: ~1757.0 H/s
- policy OFF total: ~1729.2 H/s
- observed delta: ~+27.8 H/s

Component telemetry:
- CPU ON: ~371.1 H/s
- CPU OFF: ~348.3 H/s
- CPU delta: ~+22.8 H/s
- GPU ON: ~1385.9 H/s
- GPU OFF: ~1380.9 H/s
- GPU delta: ~+5.0 H/s

The six combo rules therefore appear to be a real CPU improvement, but the practical whole-miner gain is only about +20 to +30 H/s, far below the project requirement of +100 H/s sustained.

A useful cross-check is that applying the offline-confirmed gains to the actual policy-OFF rotation mix predicts about +23.2 H/s CPU. That closely matches the observed +22.8 H/s CPU delta in the production runs. This strongly supports that the combo policy is working as designed and that its magnitude is now understood.

## Per-combination production evidence

The six experimental rules are:
- Dark + DarkLite + Lite: offline +3.77%
- Dark + Fast + Lite: offline +11.48%
- Dark + DarkLite + Turtle: offline +24.53%
- Dark + DarkLite + TurtleLite: offline +35.54%
- Dark + Turtle + TurtleLite: offline +25.39%
- DarkLite + Turtle + TurtleLite: offline +45.64%

Production coverage was uneven, but the better-covered rules generally showed positive CPU movement. Dark + DarkLite + Lite was especially clean: roughly +4% CPU production uplift, matching the offline +3.77% result closely.

The very large offline percentages do not translate to a large whole-miner gain because those combinations occupy only part of runtime. In the final policy-OFF log, the six promoted combinations accounted for about 26.7% of completed-rotation time.

## Decision

- The combo-policy mechanism is VALID.
- The combo-policy direction is NOT the missing +100 H/s.
- Preserve the six-rule policy as an optional/retained CPU optimization layer.
- Do not spend more long-run validation time trying to turn this specific six-rule set into the primary performance breakthrough.
- Future work should return to higher-ceiling structural optimization, while keeping this measured ~+23 H/s CPU gain available to stack with future GPU gains.

## Next action

Return to structural GPU/CN phase-2 work or another architectural lever with realistic >=100 H/s whole-miner potential.

Do not repeat:
- 2-lane split multiply
- dual-state / 8-lane interleave
- ordinary 4-lane 64-bit split multiply
- shuffle-first AES/T-table
- multi-GPU contention work
- tiny geometry/stagger/cache tweaks

Before implementing another multiply redesign, inspect prior cooperative 32-bit multiply work and compiled SASS/resource behavior so the next structural candidate is genuinely new.
