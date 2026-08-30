#pragma once

#include "cpu/stage_profile.h"
#include "ghostrider/ghostrider.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

namespace yerbas::cpu {

inline bool stage_profile_diagnostics_enabled()
{
    const char* value = std::getenv("YERBAS_DIAGNOSTICS");
    return value != nullptr && *value != '\0' && std::string(value) != "0";
}

inline const char* stage_profile_name(std::uint8_t stage)
{
    static constexpr const char* core_names[15] = {
        "BLAKE", "BMW", "Groestl", "JH", "Keccak",
        "Skein", "Luffa", "CubeHash", "Shavite", "SIMD",
        "Echo", "Hamsi", "Fugue", "Shabal", "Whirlpool"
    };
    if ((stage & ghostrider::kCryptoNightStageFlag) != 0)
        return ghostrider::cryptonight_name(static_cast<std::uint8_t>(stage & 0x7fU));
    return stage < 15U ? core_names[stage] : "Core?";
}

inline void sample_ghostrider_stages(const std::array<std::uint8_t,80>& header,
                                     const ghostrider::StageSchedule& schedule,
                                     std::uint64_t fingerprint)
{
    std::array<double,18> stage_ms{};
    ghostrider::Hash512 state{};

    for (std::size_t i = 0; i < schedule.size(); ++i) {
        const auto begin = std::chrono::steady_clock::now();
        const std::uint8_t stage = schedule[i];
        bool ok = false;
        ghostrider::Hash512 next{};

        if ((stage & ghostrider::kCryptoNightStageFlag) != 0) {
            ok = ghostrider::optimized_cn_stage(
                state, static_cast<std::uint8_t>(stage & 0x7fU), next);
        } else {
            const ghostrider::Work input = i == 0
                ? ghostrider::Work{header.data(), header.size()}
                : ghostrider::Work{state.data(), state.size()};
            ok = ghostrider::optimized_core_stage(input, stage, next);
        }
        if (!ok) return;
        state = next;

        const auto end = std::chrono::steady_clock::now();
        stage_ms[i] = std::chrono::duration<double, std::milli>(end - begin).count();
    }

    const auto summary = record_stage_profile(fingerprint, schedule, stage_ms);
    if (!stage_profile_diagnostics_enabled()) return;
    if (summary.samples != 1U && (summary.samples % 8U) != 0U) return;

    std::size_t hot = 0;
    std::size_t hot_core = 0;
    bool have_core = false;
    double cn_ms = 0.0;
    double core_ms = 0.0;

    for (std::size_t i = 0; i < summary.stage_ewma_ms.size(); ++i) {
        if (summary.stage_ewma_ms[i] > summary.stage_ewma_ms[hot]) hot = i;
        if ((schedule[i] & ghostrider::kCryptoNightStageFlag) != 0) {
            cn_ms += summary.stage_ewma_ms[i];
        } else {
            core_ms += summary.stage_ewma_ms[i];
            if (!have_core || summary.stage_ewma_ms[i] > summary.stage_ewma_ms[hot_core]) {
                hot_core = i;
                have_core = true;
            }
        }
    }

    const double pct = summary.total_ewma_ms > 0.0
        ? summary.stage_ewma_ms[hot] * 100.0 / summary.total_ewma_ms
        : 0.0;
    const double cn_pct = summary.total_ewma_ms > 0.0
        ? cn_ms * 100.0 / summary.total_ewma_ms : 0.0;
    const double core_pct = summary.total_ewma_ms > 0.0
        ? core_ms * 100.0 / summary.total_ewma_ms : 0.0;

    std::cout << std::fixed << std::setprecision(3)
              << "[CPU stage profile] rotation=" << std::hex << std::setw(16)
              << std::setfill('0') << fingerprint << std::dec << std::setfill(' ')
              << " | samples=" << summary.samples
              << " | total=" << summary.total_ewma_ms << " ms/hash"
              << " | CN=" << cn_ms << " ms (" << std::setprecision(1) << cn_pct << "%)"
              << std::setprecision(3)
              << " | cores=" << core_ms << " ms (" << std::setprecision(1) << core_pct << "%)"
              << std::setprecision(3)
              << " | hot=" << hot << ':' << stage_profile_name(schedule[hot])
              << ' ' << summary.stage_ewma_ms[hot] << " ms"
              << " (" << std::setprecision(1) << pct << "%)";
    if (have_core)
        std::cout << std::setprecision(3)
                  << " | hot-core=" << hot_core << ':' << stage_profile_name(schedule[hot_core])
                  << ' ' << summary.stage_ewma_ms[hot_core] << " ms";
    std::cout << std::defaultfloat << '\n';
}

} // namespace yerbas::cpu
