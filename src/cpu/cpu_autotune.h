#pragma once

#include "cpu/cpu_worker_pool.h"

#include <atomic>
#include <string>

namespace yerbas::cpu {

struct TuneResult {
    unsigned int threads{1};
    unsigned int lanes{1};
    unsigned int batch{16};
    CnWidthPolicy cn_widths{{1U, 1U, 1U, 1U, 1U, 1U}};
    double throughput_hps{0.0};
    bool from_cache{false};
    bool interrupted{false};
};

// Production CPU tuner. It first selects a compact worker/batch baseline, then
// qualifies 1/2/4-way execution independently for each CryptoNight variant.
// The final per-variant policy must pass exact batch-vs-scalar parity before it
// can be cached or used by the worker pool.
TuneResult production_autotune(unsigned int hardware_threads,
                               unsigned int configured_threads,
                               unsigned int configured_batch,
                               const std::string& mode,
                               const std::atomic_bool* stop_requested = nullptr);

} // namespace yerbas::cpu
