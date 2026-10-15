#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>

namespace yerbas::stratum {

// A one-shot, per-job experiment. Only completed, useful, unshaped scans
// qualify; a new job resets this state even when its CN triple is unchanged.
struct FastLiteGrowthState {
    unsigned int clean_scans{0};
    bool attempted{false};

    bool ready(bool eligible, bool clean_full_scan, double age_seconds)
    {
        if (!eligible || attempted || !clean_full_scan) return false;
        if (clean_scans < 3U) ++clean_scans;
        if (clean_scans < 3U || !(age_seconds >= 30.0)) return false;
        attempted = true;
        return true;
    }
};

inline std::size_t fast_lite_growth_target(std::size_t base, std::size_t safe_limit)
{
    if (base == 0U || safe_limit <= base) return base;
    const auto max = std::numeric_limits<std::size_t>::max();
    const auto extra = base / 4U;
    const auto ceiling = base > max - extra ? max : base + extra;
    // All production kernels already handle partial blocks; 128 preserves
    // alignment without requiring one whole SM group to fit in the budget.
    const auto aligned = (std::min(ceiling, safe_limit) / 128U) * 128U;
    return std::max(base, aligned);
}

} // namespace yerbas::stratum
