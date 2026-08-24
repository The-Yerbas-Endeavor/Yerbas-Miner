#pragma once

#include "config.h"

#include <atomic>

namespace yerbas {

// Shared process-wide stop request. Startup autotuners run before the Stratum
// mining loop, so they must observe the same Ctrl+C flag as production mining.
inline std::atomic_bool g_stop_requested{false};

inline bool stop_requested() noexcept
{
    return g_stop_requested.load(std::memory_order_relaxed);
}

inline void request_stop() noexcept
{
    g_stop_requested.store(true, std::memory_order_relaxed);
}

class Miner {
public:
    explicit Miner(AppConfig config);
    int run();

private:
    AppConfig config_;
};

} // namespace yerbas
