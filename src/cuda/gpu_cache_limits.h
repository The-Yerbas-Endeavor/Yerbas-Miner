#pragma once
#include <algorithm>
#include <array>
#include <cstddef>

namespace yerbas::gpu_cache {
// Reuse measured batch preferences without exceeding either the saved record
// or the current allocation. Never grow a measured batch during cache loading.
template <std::size_t N>
bool adapt_batches(std::array<std::size_t, N>& batches,
                   const std::array<std::size_t, N>& pages,
                   std::size_t saved_capacity, std::size_t saved_budget,
                   std::size_t capacity, std::size_t budget, std::size_t quantum)
{
    if (!saved_capacity || !capacity || !budget || !quantum) return false;
    auto result = batches;
    for (std::size_t i = 0; i < N; ++i) {
        const auto value = batches[i];
        const auto page = pages[i];
        if (!page || !value || value > saved_capacity ||
            (saved_budget && value > saved_budget / page) ||
            (value >= quantum && value % quantum)) return false;
        auto limit = std::min(capacity, budget / page);
        if (limit >= quantum) limit -= limit % quantum;
        if (!limit) return false;
        result[i] = std::min(value, limit);
    }
    batches = result;
    return true;
}
} // namespace yerbas::gpu_cache
