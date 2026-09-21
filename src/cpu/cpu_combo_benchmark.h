#pragma once

#include "config.h"
#include "cpu/cpu_autotune.h"
#include "cpu/cpu_worker_pool.h"
#include "ghostrider/ghostrider.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace yerbas::cpu {
namespace combo_benchmark_detail {

constexpr unsigned int kHeadersPerCombination = 3U;
constexpr double kSinglePromotionRatio = 1.015;
constexpr double kFinalPromotionRatio = 1.02;

struct MeasurementModeGuard {
    bool previous{false};
    CnWidthPolicy previous_widths{{1U, 1U, 1U, 1U, 1U, 1U}};

    MeasurementModeGuard()
        : previous(tuning_measurement_mode()), previous_widths(runtime_cn_widths())
    {
        set_tuning_measurement_mode(true);
    }

    ~MeasurementModeGuard()
    {
        set_runtime_cn_widths(previous_widths);
        set_tuning_measurement_mode(previous);
    }
};

inline bool env_enabled(const char* name)
{
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' && std::string(value) != "0";
}

inline unsigned int env_uint(const char* name, unsigned int fallback,
                             unsigned int minimum, unsigned int maximum)
{
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') return fallback;
    try {
        const unsigned long parsed = std::stoul(value);
        return static_cast<unsigned int>(
            std::max<unsigned long>(minimum, std::min<unsigned long>(maximum, parsed)));
    } catch (...) {
        return fallback;
    }
}

inline unsigned int max_width(const CnWidthPolicy& widths)
{
    return *std::max_element(widths.begin(), widths.end());
}

inline double median(std::vector<double> values)
{
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    if ((n & 1U) != 0U) return values[n / 2U];
    return 0.5 * (values[n / 2U - 1U] + values[n / 2U]);
}

inline std::string widths_string(const CnWidthPolicy& widths)
{
    std::ostringstream out;
    for (std::size_t i = 0; i < widths.size(); ++i) {
        if (i != 0U) out << '/';
        out << widths[i];
    }
    return out.str();
}

inline std::uint32_t combo_mask(const ghostrider::StageSchedule& schedule)
{
    std::uint32_t mask = 0U;
    for (const std::uint8_t stage : schedule) {
        if ((stage & ghostrider::kCryptoNightStageFlag) == 0U) continue;
        const auto variant = static_cast<unsigned int>(stage & 0x7fU);
        if (variant < 6U) mask |= (1U << variant);
    }
    return mask;
}

inline std::vector<std::size_t> active_variants(std::uint32_t mask)
{
    std::vector<std::size_t> out;
    for (std::size_t i = 0; i < 6U; ++i)
        if ((mask & (1U << i)) != 0U) out.push_back(i);
    return out;
}

inline std::string combo_name(std::uint32_t mask)
{
    std::ostringstream out;
    bool first = true;
    for (std::uint8_t i = 0; i < 6U; ++i) {
        if ((mask & (1U << i)) == 0U) continue;
        if (!first) out << '+';
        out << ghostrider::cryptonight_name(i);
        first = false;
    }
    return out.str();
}

inline std::uint64_t splitmix64(std::uint64_t& state)
{
    std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31U);
}

inline unsigned int popcount32(std::uint32_t value)
{
    unsigned int count = 0U;
    while (value != 0U) {
        value &= value - 1U;
        ++count;
    }
    return count;
}

using Header = std::array<std::uint8_t, 80>;
using HeaderSet = std::array<Header, kHeadersPerCombination>;

inline std::map<std::uint32_t, HeaderSet> discover_headers()
{
    std::map<std::uint32_t, std::vector<Header>> found;
    std::uint64_t rng = 0x594552424153434eULL; // "YERBASCN"

    for (std::uint32_t attempt = 0; attempt < 250000U; ++attempt) {
        Header header{};
        header[0] = 4U;
        for (std::size_t i = 4U; i < 36U; i += 8U) {
            const std::uint64_t value = splitmix64(rng);
            for (std::size_t b = 0; b < 8U; ++b)
                header[i + b] = static_cast<std::uint8_t>(value >> (8U * b));
        }
        for (std::size_t i = 36U; i < 76U; ++i)
            header[i] = static_cast<std::uint8_t>((attempt * 37U + i * 29U + 11U) & 0xffU);

        const ghostrider::Work work{header.data(), header.size()};
        const auto schedule = ghostrider::stage_schedule_quiet(work);
        const std::uint32_t mask = combo_mask(schedule);
        if (popcount32(mask) != 3U) continue;

        auto& bucket = found[mask];
        if (bucket.size() < kHeadersPerCombination) bucket.push_back(header);

        bool complete = found.size() == 20U;
        if (complete) {
            for (const auto& pair : found) {
                if (pair.second.size() < kHeadersPerCombination) {
                    complete = false;
                    break;
                }
            }
        }
        if (complete) break;
    }

    std::map<std::uint32_t, HeaderSet> out;
    for (const auto& pair : found) {
        if (pair.second.size() < kHeadersPerCombination) continue;
        HeaderSet headers{};
        for (std::size_t i = 0; i < headers.size(); ++i) headers[i] = pair.second[i];
        out.emplace(pair.first, headers);
    }
    return out;
}

inline void write_nonce(Header& header, std::uint32_t nonce)
{
    header[76] = static_cast<std::uint8_t>(nonce);
    header[77] = static_cast<std::uint8_t>(nonce >> 8U);
    header[78] = static_cast<std::uint8_t>(nonce >> 16U);
    header[79] = static_cast<std::uint8_t>(nonce >> 24U);
}

inline bool parity_policy(const HeaderSet& headers, const CnWidthPolicy& widths)
{
    for (std::size_t h = 0; h < headers.size(); ++h) {
        std::array<Header, 4> lane_headers{};
        std::array<ghostrider::Work, 4> works{};
        std::array<ghostrider::Hash256, 4> batch_hashes{};
        std::array<ghostrider::Hash256, 4> scalar_hashes{};

        for (std::size_t lane = 0; lane < lane_headers.size(); ++lane) {
            lane_headers[lane] = headers[h];
            write_nonce(lane_headers[lane],
                        0x6a000000U + static_cast<std::uint32_t>(h * 0x100U + lane));
            works[lane] = {lane_headers[lane].data(), lane_headers[lane].size()};
            scalar_hashes[lane] = ghostrider::hash_optimized(works[lane]);
        }

        if (!ghostrider::hash_optimized_batch(
                works.data(), batch_hashes.data(), works.size(), widths))
            return false;
        if (batch_hashes != scalar_hashes) return false;
    }
    return true;
}

inline double benchmark_policy(const HeaderSet& headers,
                               const CnWidthPolicy& widths,
                               const TuneResult& tune,
                               std::uint32_t nonce_salt)
{
    set_runtime_cn_widths(widths);
    WorkerPool pool(tune.threads, max_width(widths), tune.affinity);
    std::array<std::uint8_t, 32> impossible_target{};
    std::vector<double> samples;
    samples.reserve(headers.size());

    for (std::size_t i = 0; i < headers.size(); ++i) {
        const unsigned int warm = std::min(4U, std::max(1U, tune.batch));
        (void)pool.run(headers[i], impossible_target,
                       0x71000000U + nonce_salt + static_cast<std::uint32_t>(i * 0x10000U),
                       warm, nullptr);

        const auto begin = std::chrono::steady_clock::now();
        (void)pool.run(headers[i], impossible_target,
                       0x72000000U + nonce_salt + static_cast<std::uint32_t>(i * 0x10000U),
                       tune.batch, nullptr);
        const auto end = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(end - begin).count();
        if (seconds > 0.0)
            samples.push_back(static_cast<double>(tune.threads) * tune.batch / seconds);
    }
    return median(std::move(samples));
}

inline void add_unique(std::vector<CnWidthPolicy>& policies, const CnWidthPolicy& candidate)
{
    if (std::find(policies.begin(), policies.end(), candidate) == policies.end())
        policies.push_back(candidate);
}

struct ComboResult {
    std::uint32_t mask{0U};
    CnWidthPolicy selected{{1U, 1U, 1U, 1U, 1U, 1U}};
    double baseline_hps{0.0};
    double candidate_hps{0.0};
    double confirmed_baseline_hps{0.0};
    double confirmed_candidate_hps{0.0};
    double gain_pct{0.0};
    bool parity{false};
};

inline ComboResult benchmark_combination(std::uint32_t mask,
                                         const HeaderSet& headers,
                                         const TuneResult& tune,
                                         unsigned int confirm_passes)
{
    ComboResult result{};
    result.mask = mask;
    result.selected = tune.cn_widths;

    std::vector<CnWidthPolicy> candidates;
    add_unique(candidates, tune.cn_widths);

    const auto active = active_variants(mask);
    constexpr std::array<unsigned int, 3> widths{{1U, 2U, 4U}};
    for (const std::size_t variant : active) {
        for (const unsigned int width : widths) {
            if (width == tune.cn_widths[variant]) continue;
            CnWidthPolicy candidate = tune.cn_widths;
            candidate[variant] = width;
            add_unique(candidates, candidate);
        }
    }

    std::map<std::string, double> measured;
    std::array<unsigned int, 6> promising_width{{0U, 0U, 0U, 0U, 0U, 0U}};
    std::array<bool, 6> promising{{false, false, false, false, false, false}};

    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const auto& candidate = candidates[i];
        if (!parity_policy(headers, candidate)) continue;
        const double hps = benchmark_policy(
            headers, candidate, tune, static_cast<std::uint32_t>(i * 0x1000U));
        measured[widths_string(candidate)] = hps;
        if (candidate == tune.cn_widths) result.baseline_hps = hps;
    }

    for (const std::size_t variant : active) {
        double best = result.baseline_hps;
        unsigned int best_width = tune.cn_widths[variant];
        for (const unsigned int width : widths) {
            if (width == tune.cn_widths[variant]) continue;
            CnWidthPolicy candidate = tune.cn_widths;
            candidate[variant] = width;
            const auto it = measured.find(widths_string(candidate));
            if (it != measured.end() && it->second > best) {
                best = it->second;
                best_width = width;
            }
        }
        if (result.baseline_hps > 0.0 &&
            best_width != tune.cn_widths[variant] &&
            best >= result.baseline_hps * kSinglePromotionRatio) {
            promising[variant] = true;
            promising_width[variant] = best_width;
        }
    }

    std::vector<std::size_t> p;
    for (std::size_t i = 0; i < promising.size(); ++i)
        if (promising[i]) p.push_back(i);

    for (std::size_t i = 0; i < p.size(); ++i) {
        for (std::size_t j = i + 1U; j < p.size(); ++j) {
            CnWidthPolicy candidate = tune.cn_widths;
            candidate[p[i]] = promising_width[p[i]];
            candidate[p[j]] = promising_width[p[j]];
            add_unique(candidates, candidate);
        }
    }
    if (p.size() >= 3U) {
        CnWidthPolicy candidate = tune.cn_widths;
        for (const std::size_t variant : p) candidate[variant] = promising_width[variant];
        add_unique(candidates, candidate);
    }

    for (std::size_t i = 0; i < candidates.size(); ++i) {
        const auto& candidate = candidates[i];
        const std::string key = widths_string(candidate);
        if (measured.find(key) != measured.end()) continue;
        if (!parity_policy(headers, candidate)) continue;
        measured[key] = benchmark_policy(
            headers, candidate, tune, static_cast<std::uint32_t>((i + 16U) * 0x1000U));
    }

    result.candidate_hps = result.baseline_hps;
    for (const auto& candidate : candidates) {
        const auto it = measured.find(widths_string(candidate));
        if (it == measured.end()) continue;
        if (it->second > result.candidate_hps) {
            result.candidate_hps = it->second;
            result.selected = candidate;
        }
    }

    result.parity = parity_policy(headers, result.selected);
    if (!result.parity || result.baseline_hps <= 0.0 ||
        result.candidate_hps < result.baseline_hps * kFinalPromotionRatio) {
        result.selected = tune.cn_widths;
        result.candidate_hps = result.baseline_hps;
        result.confirmed_baseline_hps = result.baseline_hps;
        result.confirmed_candidate_hps = result.baseline_hps;
        result.gain_pct = 0.0;
        return result;
    }

    std::vector<double> base_samples;
    std::vector<double> candidate_samples;
    base_samples.reserve(confirm_passes);
    candidate_samples.reserve(confirm_passes);

    for (unsigned int pass = 0; pass < confirm_passes; ++pass) {
        const std::uint32_t salt = static_cast<std::uint32_t>(0x200000U + pass * 0x20000U);
        if ((pass & 1U) == 0U) {
            base_samples.push_back(benchmark_policy(headers, tune.cn_widths, tune, salt));
            candidate_samples.push_back(benchmark_policy(headers, result.selected, tune, salt + 0x10000U));
        } else {
            candidate_samples.push_back(benchmark_policy(headers, result.selected, tune, salt));
            base_samples.push_back(benchmark_policy(headers, tune.cn_widths, tune, salt + 0x10000U));
        }
    }

    result.confirmed_baseline_hps = median(std::move(base_samples));
    result.confirmed_candidate_hps = median(std::move(candidate_samples));
    if (result.confirmed_baseline_hps > 0.0) {
        result.gain_pct =
            (result.confirmed_candidate_hps / result.confirmed_baseline_hps - 1.0) * 100.0;
    }
    return result;
}

} // namespace combo_benchmark_detail

inline bool cpu_combo_benchmark_requested()
{
    return combo_benchmark_detail::env_enabled("YERBAS_CPU_COMBO_BENCH");
}

inline int run_cpu_cn_combo_benchmark(const AppConfig& config)
{
    using namespace combo_benchmark_detail;

    if (!config.miner.cpu_enabled) {
        std::cerr << "CPU combo benchmark requires CPU mining to be enabled.\n";
        return 2;
    }

    const unsigned int hw_threads = std::max(1U, std::thread::hardware_concurrency());
    const TuneResult tune = production_autotune(
        hw_threads, config.miner.threads, config.miner.cpu_batch,
        config.miner.cpu_tune, nullptr);

    if (tune.interrupted) return 130;
    if (!tune.from_cache) {
        std::cerr << "CPU combo benchmark requires an existing cached CPU policy.\n"
                  << "Run normal CPU tuning first; refusing to benchmark after creating a new baseline.\n";
        return 3;
    }

    MeasurementModeGuard guard;
    set_runtime_lane_width(tune.lanes);
    set_runtime_cn_widths(tune.cn_widths);
    set_runtime_affinity_policy(tune.affinity);

    const unsigned int confirm_passes =
        env_uint("YERBAS_CPU_COMBO_PASSES", 5U, 3U, 9U);

    std::cout << "\n============================================================\n"
              << " Yerbas CPU CryptoNight combination benchmark\n"
              << "============================================================\n"
              << "Baseline workers : " << tune.threads << '\n'
              << "Baseline batch   : " << tune.batch << '\n'
              << "Baseline widths  : " << widths_string(tune.cn_widths) << '\n'
              << "Affinity         : " << affinity_policy_name(tune.affinity) << '\n'
              << "Confirm passes   : " << confirm_passes << '\n'
              << "Policy           : diagnostic only; no production cache writes\n\n";

    const auto header_sets = discover_headers();
    if (header_sets.size() != 20U) {
        std::cerr << "ERROR: found only " << header_sets.size()
                  << " of 20 three-CN combinations.\n";
        return 4;
    }

    std::vector<ComboResult> results;
    results.reserve(header_sets.size());

    std::size_t index = 0U;
    for (const auto& pair : header_sets) {
        ++index;
        std::cout << "[CPU combo] " << index << "/20 | " << combo_name(pair.first)
                  << " | testing..." << std::endl;
        const ComboResult result =
            benchmark_combination(pair.first, pair.second, tune, confirm_passes);
        results.push_back(result);

        std::cout << std::fixed << std::setprecision(2)
                  << "[CPU combo result] " << combo_name(result.mask)
                  << " | base=" << result.confirmed_baseline_hps << " H/s"
                  << " | selected=" << result.confirmed_candidate_hps << " H/s"
                  << " | gain=" << (result.gain_pct >= 0.0 ? "+" : "")
                  << result.gain_pct << "%"
                  << " | widths=" << widths_string(result.selected)
                  << " | parity=" << (result.parity ? "PASS" : "FAIL")
                  << std::defaultfloat << '\n';
    }

    double equal_combo_base = 0.0;
    double equal_combo_selected = 0.0;
    unsigned int promoted = 0U;
    for (const auto& result : results) {
        equal_combo_base += result.confirmed_baseline_hps;
        equal_combo_selected += result.confirmed_candidate_hps;
        if (result.gain_pct >= 2.0 && result.selected != tune.cn_widths) ++promoted;
    }
    equal_combo_base /= static_cast<double>(results.size());
    equal_combo_selected /= static_cast<double>(results.size());
    const double overall_gain = equal_combo_base > 0.0
        ? (equal_combo_selected / equal_combo_base - 1.0) * 100.0 : 0.0;

    std::cout << std::fixed << std::setprecision(2)
              << "\n================ CPU COMBO SUMMARY ================\n"
              << "Combinations tested : " << results.size() << "/20\n"
              << "Promoted candidates : " << promoted << '\n'
              << "Equal-combo baseline: " << equal_combo_base << " H/s\n"
              << "Equal-combo selected: " << equal_combo_selected << " H/s\n"
              << "Projected CPU gain  : " << (overall_gain >= 0.0 ? "+" : "")
              << overall_gain << "%\n"
              << "No production policy or cache was modified.\n"
              << "===================================================\n"
              << std::defaultfloat;

    return 0;
}

} // namespace yerbas::cpu
