#pragma once

#include <atomic>
#include <string>

namespace yerbas::cpu {

struct TuneResult {
    unsigned int threads{1};
    unsigned int lanes{1};
    unsigned int batch{16};
    double throughput_hps{0.0};
    bool from_cache{false};
    bool interrupted{false};
};

// Single production CPU tuner. Benchmarks complete GhostRider work and selects
// workers + lane width + per-worker batch as one policy. Old split thread/batch
// and lane-scheduler tuners are intentionally retired.
TuneResult production_autotune(unsigned int hardware_threads,
                               unsigned int configured_threads,
                               unsigned int configured_batch,
                               const std::string& mode,
                               const std::atomic_bool* stop_requested = nullptr);

} // namespace yerbas::cpu
