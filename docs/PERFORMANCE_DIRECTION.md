# CUDA production performance direction

The CN-Fast v4-v9 experiment chain was useful for ruling out several narrow loop rewrites, but it did not produce a repeatable whole-miner improvement large enough to justify carrying the extra production paths.

Production optimization now follows two rules:

1. Prefer changes that improve complete GhostRider throughput, not isolated kernel microbenchmarks.
2. Keep hardware selection generic and runtime-tuned; do not hard-code a GPU model or architecture.

## Proven CN-Fast result

The setup/coop/final geometry tuner remains because it produced a parity-qualified whole-CN-Fast improvement greater than 5% on one device and safely retained baseline geometry on another. Its cache is keyed by device, driver/runtime, and active batch size.

## Next high-impact target

The 15 conventional GhostRider core stages currently use a fixed CUDA launch width. The next production tuner should benchmark legal launch widths per core algorithm, cache the per-device winners, and qualify changes using end-to-end GhostRider hashes/second. This affects a much larger fraction of the 18-stage GhostRider pipeline than another CN-Fast-only rewrite.

Candidate launch widths should remain generic (for example 64/128/256/512 where supported), parity must remain exact, and a winner should be accepted only after a repeated whole-pipeline throughput improvement rather than a single kernel timing.
