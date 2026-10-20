# CPU batch work sharing

Persistent CPU workers take four-hash chunks from a shared nonce cursor. A worker that finishes early can continue computing instead of waiting for another worker to finish a fixed slice. Configured batch work and the tuner-selected active worker count remain unchanged. Small batches and single-worker runs use fixed slices. The batch selects one scheduling mode for all active workers, including uneven divisions.

This is generic scheduling: there are no CPU-model rules or native compiler flags. Existing hash implementations, runtime instruction dispatch, lane policy, cancellation and actual-completed-hash accounting remain in use. Cursor claims use relaxed atomic operations; existing batch publication and completion mutexes synchronize metadata and results.

Work sharing is enabled by default. Set `YERBAS_CPU_WORKSHARE=0` before starting the process to compare fixed slices. The option is captured when the worker pool is constructed.

Correctness checks cover nonce uniqueness and complete coverage, uint32 nonce wrap, zero and tiny batches, partial lane groups, inactive workers, uneven active-worker allocations and interrupted batches. Performance should be compared on identical inputs with alternating test order; CPU throughput is not whole-miner or pool-effective throughput.
