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

The policy-ON run did not demonstrate the required +100 H/s whole-miner gain versus the established ~1.86-1.87 kH/s stable reference.

## Policy-OFF baseline run

Log snapshot: `cpu-combo-production-baseline-20260917-092041.log`

Same already-built binary, combo policy OFF.

Startup is clean and comparable on CPU:
- workers=6
- batch=16
- widths=1/4/1/2/4/1
- affinity=unpinned
- cached throughput=432.70 H/s

Both GPUs started from populated calibration/variant caches.

Current uploaded snapshot covers only about 29 minutes, so it is NOT sufficient for a whole-miner long-run verdict.

Snapshot telemetry:
- ~17 completed rotations
- ~0.485h completed-rotation time
- exact weighted telemetry: ~1693.2 H/s total, ~378.1 H/s CPU, ~1315.1 H/s GPU
- latest status near 28m: miner AVG ~1.78 kH/s, 324 accepted, 1 rejected, 1 block

## Early same-combination signal

The short baseline already overlaps two of the six policy combinations.

For Dark + DarkLite + Lite (policy mask 11):
- policy ON, completed rotations >=20s: ~533.6 H/s CPU across ~1618s
- policy OFF: ~515.6 H/s CPU across ~814s
- observed CPU delta: about +3.5% with policy ON

This closely matches the offline benchmark's confirmed +3.77% rule, which is useful evidence that the policy hook is actually applying a real gain.

For Dark + Fast + Lite (policy mask 13):
- policy ON: ~266.9 H/s CPU across ~2892s
- policy OFF: ~203.9 H/s CPU across only ~96s
- observed CPU delta is large (~+31%), but the OFF sample is too short to trust yet.

The other four promoted combinations have not appeared enough in the short OFF snapshot to validate them.

## Immediate action

If the policy-OFF miner is still running, leave it running in the foreground. Do not rebuild or retune. Capture/upload a later snapshot or final log after several more hours so all six policy combinations have meaningful OFF samples.

Do not merge or kill the combo policy until the same-binary ON/OFF comparison has enough per-combination coverage.
