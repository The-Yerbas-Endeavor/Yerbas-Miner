#include "cpu/cpu_autotune.h"

#include "cpu/cpu_worker_pool.h"
#include "ghostrider/ghostrider.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace yerbas::cpu {
namespace {

constexpr int kCpuPolicyRevision = 3;
constexpr double kWidthRequiredGain = 1.02;

bool env_enabled(const char* name)
{
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' && std::string(value) != "0";
}

bool stopped(const std::atomic_bool* stop)
{
    return stop != nullptr && stop->load(std::memory_order_relaxed);
}

struct TuningMeasurementGuard {
    TuningMeasurementGuard() { set_tuning_measurement_mode(true); }
    ~TuningMeasurementGuard() { set_tuning_measurement_mode(false); }
};

std::filesystem::path cache_dir()
{
#ifdef _WIN32
    if (const char* p = std::getenv("LOCALAPPDATA")) return std::filesystem::path(p) / "Yerbas-Miner" / "cache";
    if (const char* p = std::getenv("USERPROFILE")) return std::filesystem::path(p) / ".cache" / "yerbas-miner";
#else
    if (const char* p = std::getenv("XDG_CACHE_HOME")) return std::filesystem::path(p) / "yerbas-miner";
    if (const char* p = std::getenv("HOME")) return std::filesystem::path(p) / ".cache" / "yerbas-miner";
#endif
    return std::filesystem::path(".") / ".yerbas-miner-cache";
}

std::filesystem::path cache_path(unsigned int hardware_threads,
                                 unsigned int ceiling,
                                 const std::string& mode)
{
    std::string build = "portable";
#ifdef YERBAS_NATIVE_CPU_BUILD
    build = "native";
#endif
    return cache_dir() / ("cpu-policy-rev" + std::to_string(kCpuPolicyRevision) +
                          "-hw" + std::to_string(hardware_threads) +
                          "-max" + std::to_string(ceiling) +
                          "-" + build + "-" + mode + ".txt");
}

std::array<std::array<std::uint8_t, 80>, 3> representative_headers()
{
    std::array<std::array<std::uint8_t, 80>, 3> headers{};
    for (std::size_t h = 0; h < headers.size(); ++h) {
        headers[h][0] = 4;
        for (std::size_t i = 4; i < 76; ++i)
            headers[h][i] = static_cast<std::uint8_t>((i * (17U + h * 8U) + 31U + h * 37U) & 0xffU);
    }
    return headers;
}

std::array<std::array<std::uint8_t, 80>, 6> headers_by_cn_variant()
{
    std::array<std::array<std::uint8_t, 80>, 6> headers{};
    std::array<bool, 6> found{};
    std::size_t remaining = found.size();

    for (unsigned int seed = 0; seed < 512U && remaining > 0; ++seed) {
        std::array<std::uint8_t, 80> header{};
        header[0] = 4;
        for (std::size_t i = 4; i < 76; ++i)
            header[i] = static_cast<std::uint8_t>((i * (19U + seed * 2U) + 43U + seed * 29U) & 0xffU);
        const ghostrider::Work work{header.data(), header.size()};
        const auto schedule = ghostrider::stage_schedule_quiet(work);
        for (const auto encoded : schedule) {
            if ((encoded & ghostrider::kCryptoNightStageFlag) == 0) continue;
            const auto variant = static_cast<std::size_t>(encoded & 0x7fU);
            if (variant >= found.size() || found[variant]) continue;
            headers[variant] = header;
            found[variant] = true;
            --remaining;
        }
    }

    if (remaining != 0) {
        const auto fallback = representative_headers();
        for (std::size_t i = 0; i < found.size(); ++i)
            if (!found[i]) headers[i] = fallback[i % fallback.size()];
    }
    return headers;
}

struct Plan {
    unsigned int workers{1};
    unsigned int batch{16};
};

std::vector<unsigned int> batches_for(const std::string& mode, unsigned int configured)
{
    std::set<unsigned int> values;
    if (mode == "full") values = {8U, 12U, 16U, 20U, 24U, 28U, 32U, 40U, 48U, 64U};
    else values = {16U, 32U};
    if (configured > 0U && configured <= 128U) values.insert(configured);
    return {values.begin(), values.end()};
}

std::vector<Plan> plans_for(unsigned int ceiling,
                            unsigned int configured_batch,
                            const std::string& mode)
{
    std::vector<Plan> out;
    const auto batches = batches_for(mode, configured_batch);
    std::set<unsigned int> workers;
    if (mode == "full") {
        for (unsigned int w = 1; w <= ceiling; ++w) workers.insert(w);
    } else {
        workers.insert(ceiling);
        if (ceiling > 1U) workers.insert(ceiling - 1U);
        workers.insert(std::max(1U, ceiling / 2U));
    }
    for (const auto w : workers)
        for (const auto batch : batches)
            out.push_back({w, batch});
    return out;
}

unsigned int max_width(const CnWidthPolicy& widths)
{
    return *std::max_element(widths.begin(), widths.end());
}

double benchmark_header(const Plan& plan,
                        const CnWidthPolicy& widths,
                        const std::array<std::uint8_t, 80>& header,
                        const std::atomic_bool* stop)
{
    if (stopped(stop)) return 0.0;
    TuningMeasurementGuard measurement_guard;
    set_runtime_cn_widths(widths);
    const unsigned int group = max_width(widths);
    WorkerPool pool(plan.workers, group);
    std::array<std::uint8_t, 32> impossible_target{};

    (void)pool.run(header, impossible_target, 0x51000000U,
                   std::min(4U, std::max(1U, plan.batch)));
    const auto begin = std::chrono::steady_clock::now();
    (void)pool.run(header, impossible_target, 0x52000000U, plan.batch);
    const auto end = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(end - begin).count();
    return seconds > 0.0 ? static_cast<double>(plan.workers) * plan.batch / seconds : 0.0;
}

double benchmark_plan(const Plan& plan,
                      const CnWidthPolicy& widths,
                      const std::atomic_bool* stop)
{
    const auto headers = representative_headers();
    std::vector<double> samples;
    samples.reserve(headers.size());
    for (const auto& header : headers) {
        if (stopped(stop)) return 0.0;
        const double hps = benchmark_header(plan, widths, header, stop);
        if (hps > 0.0) samples.push_back(hps);
    }
    if (samples.empty()) return 0.0;
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2U];
}

bool parity_width_policy(const CnWidthPolicy& widths)
{
    const auto headers = representative_headers();
    for (std::size_t sample = 0; sample < headers.size(); ++sample) {
        std::array<std::array<std::uint8_t, 80>, 4> lane_headers{};
        std::array<ghostrider::Work, 4> works{};
        std::array<ghostrider::Hash256, 4> batch_hashes{};
        std::array<ghostrider::Hash256, 4> scalar_hashes{};
        for (std::size_t lane = 0; lane < lane_headers.size(); ++lane) {
            lane_headers[lane] = headers[sample];
            const std::uint32_t nonce = 0x63000000U + static_cast<std::uint32_t>(sample * 0x100U + lane);
            lane_headers[lane][76] = static_cast<std::uint8_t>(nonce);
            lane_headers[lane][77] = static_cast<std::uint8_t>(nonce >> 8);
            lane_headers[lane][78] = static_cast<std::uint8_t>(nonce >> 16);
            lane_headers[lane][79] = static_cast<std::uint8_t>(nonce >> 24);
            works[lane] = {lane_headers[lane].data(), lane_headers[lane].size()};
            scalar_hashes[lane] = ghostrider::hash_optimized(works[lane]);
        }
        if (!ghostrider::hash_optimized_batch(works.data(), batch_hashes.data(), works.size(), widths))
            return false;
        if (batch_hashes != scalar_hashes) return false;
    }
    return true;
}

bool valid_width(unsigned int width)
{
    return width == 1U || width == 2U || width == 4U;
}

bool load_cache(const std::filesystem::path& path, TuneResult& out)
{
    if (env_enabled("YERBAS_CPU_RETUNE")) return false;
    std::ifstream in(path);
    std::string magic;
    int revision = 0;
    if (!(in >> magic >> revision >> out.threads >> out.lanes >> out.batch)) return false;
    for (auto& width : out.cn_widths)
        if (!(in >> width)) return false;
    if (!(in >> out.throughput_hps)) return false;
    if (magic != "YERBAS_CPU_POLICY" || revision != kCpuPolicyRevision ||
        out.threads == 0U || out.batch == 0U || !valid_width(out.lanes)) return false;
    for (const auto width : out.cn_widths)
        if (!valid_width(width)) return false;
    if (out.lanes != max_width(out.cn_widths)) return false;
    out.from_cache = true;
    return true;
}

void save_cache(const std::filesystem::path& path, const TuneResult& value)
{
    try {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream out(path, std::ios::trunc);
        if (!out) return;
        out << "YERBAS_CPU_POLICY " << kCpuPolicyRevision << ' '
            << value.threads << ' ' << value.lanes << ' ' << value.batch;
        for (const auto width : value.cn_widths) out << ' ' << width;
        out << ' ' << std::setprecision(12) << value.throughput_hps << '\n';
    } catch (...) {}
}

bool load_compatible_cache(unsigned int hardware_threads,
                           unsigned int ceiling,
                           const std::string& mode,
                           TuneResult& out,
                           unsigned int& source_ceiling)
{
    if (env_enabled("YERBAS_CPU_RETUNE")) return false;

    bool found = false;
    TuneResult best{};
    unsigned int best_source = ceiling;

    for (unsigned int candidate_ceiling = ceiling;
         candidate_ceiling <= hardware_threads;
         ++candidate_ceiling) {
        TuneResult candidate{};
        if (!load_cache(cache_path(hardware_threads, candidate_ceiling, mode), candidate))
            continue;
        if (candidate.threads > ceiling)
            continue;
        if (!parity_width_policy(candidate.cn_widths))
            continue;
        if (!found || candidate.throughput_hps > best.throughput_hps) {
            best = candidate;
            best_source = candidate_ceiling;
            found = true;
        }
    }

    if (!found) return false;
    out = best;
    source_ceiling = best_source;
    return true;
}

} // namespace

TuneResult production_autotune(unsigned int hardware_threads,
                               unsigned int configured_threads,
                               unsigned int configured_batch,
                               const std::string& mode,
                               const std::atomic_bool* stop)
{
    hardware_threads = std::max(1U, hardware_threads);
    const unsigned int ceiling = configured_threads == 0U
        ? hardware_threads
        : std::max(1U, std::min(configured_threads, hardware_threads));
    const unsigned int fallback_batch = configured_batch == 0U ? 16U : configured_batch;
    const CnWidthPolicy scalar_widths{{1U, 1U, 1U, 1U, 1U, 1U}};

    if (mode == "off" || env_enabled("YERBAS_CPU_DISABLE_AUTOTUNE"))
        return TuneResult{ceiling, 1U, fallback_batch, scalar_widths, 0.0, false, false};

    TuneResult cached{};
    const auto path = cache_path(hardware_threads, ceiling, mode);
    unsigned int source_ceiling = ceiling;
    if (load_compatible_cache(hardware_threads, ceiling, mode, cached, source_ceiling)) {
        set_runtime_cn_widths(cached.cn_widths);
        if (source_ceiling != ceiling)
            save_cache(path, cached);
        std::cout << "[CPU tune] cached | mode=" << mode
                  << " | workers=" << cached.threads
                  << " | batch=" << cached.batch
                  << " | CN widths=" << cached.cn_widths[0] << '/' << cached.cn_widths[1] << '/'
                  << cached.cn_widths[2] << '/' << cached.cn_widths[3] << '/'
                  << cached.cn_widths[4] << '/' << cached.cn_widths[5]
                  << " | throughput=" << std::fixed << std::setprecision(2)
                  << cached.throughput_hps << " H/s"
                  << " | source=max" << source_ceiling
                  << (source_ceiling == ceiling ? "" : " (compatible)")
                  << std::defaultfloat << '\n';
        return cached;
    }

    const auto plans = plans_for(ceiling, fallback_batch, mode);
    std::cout << "[CPU tune] GhostRider production search"
              << " | mode=" << mode
              << " | hardware_threads=" << hardware_threads
              << " | ceiling=" << ceiling
              << " | baseline_candidates=" << plans.size()
              << " | CN-width-probes=6x(1/2/4)"
              << " | metric=end-to-end H/s\n";

    TuneResult best{ceiling, 1U, fallback_batch, scalar_widths, 0.0, false, false};
    for (const auto& plan : plans) {
        if (stopped(stop)) {
            best.interrupted = true;
            return best;
        }
        const double hps = benchmark_plan(plan, scalar_widths, stop);
        if (env_enabled("YERBAS_DIAGNOSTICS")) {
            std::cout << "[CPU baseline] workers=" << plan.workers
                      << " | batch=" << plan.batch
                      << " | " << std::fixed << std::setprecision(2) << hps
                      << " H/s" << std::defaultfloat << '\n';
        }
        if (hps > best.throughput_hps) {
            best.threads = plan.workers;
            best.batch = plan.batch;
            best.throughput_hps = hps;
        }
    }

    if (stopped(stop)) {
        best.interrupted = true;
        return best;
    }

    const Plan selected_plan{best.threads, best.batch};
    const auto variant_headers = headers_by_cn_variant();
    CnWidthPolicy widths = scalar_widths;

    for (std::size_t variant = 0; variant < widths.size(); ++variant) {
        if (stopped(stop)) {
            best.interrupted = true;
            return best;
        }
        const double baseline_hps = benchmark_header(selected_plan, widths, variant_headers[variant], stop);
        unsigned int selected_width = widths[variant];
        double selected_hps = baseline_hps;
        for (const unsigned int candidate_width : {2U, 4U}) {
            CnWidthPolicy candidate = widths;
            candidate[variant] = candidate_width;
            const double hps = benchmark_header(selected_plan, candidate, variant_headers[variant], stop);
            if (hps > selected_hps * kWidthRequiredGain) {
                selected_hps = hps;
                selected_width = candidate_width;
            }
        }
        widths[variant] = selected_width;
        std::cout << "[CPU CN width] " << ghostrider::cryptonight_name(static_cast<std::uint8_t>(variant))
                  << " | selected=" << selected_width
                  << " | baseline=" << std::fixed << std::setprecision(2) << baseline_hps
                  << " H/s | best=" << selected_hps << " H/s"
                  << std::defaultfloat << '\n';
    }

    const double tuned_hps = benchmark_plan(selected_plan, widths, stop);
    const bool parity_ok = parity_width_policy(widths);
    const bool useful = parity_ok && tuned_hps > best.throughput_hps * 1.01;
    if (useful) {
        best.cn_widths = widths;
        best.lanes = max_width(widths);
        best.throughput_hps = tuned_hps;
    } else {
        best.cn_widths = scalar_widths;
        best.lanes = 1U;
    }

    set_runtime_cn_widths(best.cn_widths);
    save_cache(path, best);
    std::cout << "[CPU tune] selected"
              << " | workers=" << best.threads
              << " | batch=" << best.batch
              << " | CN widths=" << best.cn_widths[0] << '/' << best.cn_widths[1] << '/'
              << best.cn_widths[2] << '/' << best.cn_widths[3] << '/'
              << best.cn_widths[4] << '/' << best.cn_widths[5]
              << " | parity=" << (parity_ok ? "PASS" : "FAIL-scalar")
              << " | throughput=" << std::fixed << std::setprecision(2)
              << best.throughput_hps << " H/s"
              << " | cache=cpu-policy-v3" << std::defaultfloat << '\n';
    return best;
}

} // namespace yerbas::cpu
