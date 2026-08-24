#pragma once

#include <atomic>
#include <cstdint>

namespace yerbas::cpu {

struct TuneResult {
    unsigned int threads{1};
    unsigned int batch{16};
    double throughput_hps{0.0};
    bool from_cache{false};
    bool interrupted{false};
};

// Benchmarks representative GhostRider schedules and selects a production CPU
// thread count + per-thread batch size. Results are cached per CPU/build so
// normal launches do not repeat the benchmark unless YERBAS_CPU_RETUNE is set.
// stop_requested may be supplied by the miner so Ctrl+C can abort tuning cleanly.
TuneResult production_autotune(unsigned int hardware_threads,
                               unsigned int configured_threads,
                               unsigned int configured_batch,
                               const std::atomic_bool* stop_requested = nullptr);

} // namespace yerbas::cpu
