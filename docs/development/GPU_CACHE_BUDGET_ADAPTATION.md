# GPU cache budget adaptation

This development change preserves measured class and variant preferences when startup allocation capacity changes. Existing cache lookup already searches compatible capacities. The observed miss instead comes from rejecting an entire calibration when one class exceeds the current byte budget.

The loader now validates saved limits and reduces only batches that exceed current capacity or per-class byte limits, retaining warp/SM alignment. It never increases a saved preference. Variant sidecars must match the adapted class policy and satisfy both saved and current limits. Driver, CUDA runtime, device identity and revision matching remain unchanged. Loading performs no calibration and writes no cache files. Adjusted batches are logged explicitly; their throughput is not assumed to match the original measurements.

The five-worker CPU experiment is excluded from this source snapshot. CPU hash arithmetic, GPU kernels, nonce scheduling and generic architecture defaults are unchanged. The generic CUDA build, four correctness checks, and host cache-loader integration checks passed. Live throughput qualification and Windows compilation remain required before release promotion.
