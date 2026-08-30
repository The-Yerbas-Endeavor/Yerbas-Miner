#pragma once

#include "cpu/cpu_topology.h"
#include "cpu/cpu_worker_pool.h"

#include <atomic>
#include <string>

namespace yerbas::cpu {

struct TuneResult {
    unsigned int threads{1};
    unsigned int lanes{1};
    unsigned int batch{16};
    CnWidthPolicy cn_widths{{1U, 1U, 1U, 1U, 1U, 1U}};
    AffinityPolicy affinity{AffinityPolicy::Unpinned};
    double throughput_hps{0.0};
    bool from_cache{false};
    bool interrupted{false};
};

// Production CPU tuner. It selects worker/batch baseline, qualifies scalar vs
// genuine 2-way execution independently for every CryptoNight variant using
// repeated measurements + exact parity, then benchmarks Linux topology policy.
// The winning widths and affinity policy are cached together.
TuneResult production_autotune(unsigned int hardware_threads,
                               unsigned int configured_threads,
                               unsigned int configured_batch,
                               const std::string& mode,
                               const std::atomic_bool* stop_requested = nullptr);

} // namespace yerbas::cpu
