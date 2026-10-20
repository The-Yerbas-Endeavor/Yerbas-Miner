# Shutdown hash accounting

CPU worker batches now expose the number of hashes actually completed. Cancelled batches credit zero; interrupted batches credit only finished scalar hashes or completed groups. Both live and rotation counters use this number, and worker diagnostics derive throughput from completed work.

The hybrid scheduler stops CPU dispatch after shutdown is requested, waits without a busy CPU retry loop for pending GPU results, and avoids launching replacement GPU scans. CPU-only and GPU-only entry points also honor the stop request. This prevents cancellation from inflating current, rolling, rotation and session hash rates. No speed-based clamp is used. Hash kernels and generic build settings are unchanged.

Checks exercise complete batches and nonce coverage with1,2,6 CPU workers and1,2,4 lanes,2000 pre-cancelled batches, and mid-batch interruption with all three lane widths. A full hybrid stop with physical CUDA devices still requires runtime confirmation.
