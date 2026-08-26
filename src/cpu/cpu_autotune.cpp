#include "cpu/cpu_autotune.h"

#include "cpu/cpu_worker_pool.h"

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
#include <tuple>
#include <vector>

namespace yerbas::cpu {
namespace {

constexpr int kCpuPolicyRevision = 2;

bool env_enabled(const char* name)
{
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' && std::string(value) != "0";
}

bool stopped(const std::atomic_bool* stop)
{
    return stop != nullptr && stop->load(std::memory_order_relaxed);
}

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

struct Plan {
    unsigned int workers{1};
    unsigned int lanes{1};
    unsigned int batch{16};
};

std::vector<unsigned int> batches_for(const std::string& mode, unsigned int configured)
{
    std::set<unsigned int> values;
    if (mode == "full")
        values = {8U, 12U, 16U, 20U, 24U, 28U, 32U, 40U, 48U, 64U};
    else if (mode == "simple")
        values = {16U, 32U};
    else
        values = {16U, 32U};
    if (configured > 0U && configured <= 128U) values.insert(configured);
    return {values.begin(), values.end()};
}

std::vector<Plan> plans_for(unsigned int ceiling,
                            unsigned int configured_batch,
                            const std::string& mode)
{
    std::set<std::tuple<unsigned int, unsigned int, unsigned int>> unique;
    const auto batches = batches_for(mode, configured_batch);

    auto add_policy = [&](unsigned int workers, unsigned int lanes) {
        if (workers == 0U || lanes == 0U || workers * lanes > ceiling) return;
        for (const unsigned int batch : batches)
            unique.emplace(workers, lanes, batch);
    };

    if (mode == "full") {
        for (const unsigned int lanes : {1U, 2U, 4U}) {
            if (lanes > ceiling) continue;
            const unsigned int max_workers = std::max(1U, ceiling / lanes);
            for (unsigned int workers = 1; workers <= max_workers; ++workers)
                add_policy(workers, lanes);
        }
    } else {
        // Bounded generic production search. Cover full logical occupancy,
        // one lower standard policy, and the useful 2/4-lane families without
        // walking the entire matrix at every fresh install.
        add_policy(ceiling, 1U);
        if (ceiling > 1U) add_policy(ceiling - 1U, 1U);
        add_policy(std::max(1U, ceiling / 2U), 1U);
        if (ceiling >= 2U) add_policy(std::max(1U, ceiling / 2U), 2U);
        if (ceiling >= 4U) add_policy(std::max(1U, ceiling / 4U), 4U);
    }

    std::vector<Plan> plans;
    plans.reserve(unique.size());
    for (const auto& [workers, lanes, batch] : unique)
        plans.push_back({workers, lanes, batch});
    return plans;
}

double benchmark_plan(const Plan& plan, const std::atomic_bool* stop)
{
    if (stopped(stop)) return 0.0;
    WorkerPool pool(plan.workers, plan.lanes);
    const auto headers = representative_headers();
    std::array<std::uint8_t, 32> impossible_target{};

    (void)pool.run(headers.front(), impossible_target, 0x51000000U,
                   std::min(4U, std::max(1U, plan.batch)));

    std::vector<double> samples;
    samples.reserve(headers.size());
    for (std::size_t i = 0; i < headers.size(); ++i) {
        if (stopped(stop)) return 0.0;
        const auto start = std::chrono::steady_clock::now();
        (void)pool.run(headers[i], impossible_target,
                       0x52000000U + static_cast<std::uint32_t>(i * 0x10000U),
                       plan.batch);
        const auto end = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(end - start).count();
        if (seconds > 0.0)
            samples.push_back(static_cast<double>(plan.workers) * plan.batch / seconds);
    }
    if (samples.empty()) return 0.0;
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2U];
}

bool load_cache(const std::filesystem::path& path, TuneResult& out)
{
    if (env_enabled("YERBAS_CPU_RETUNE")) return false;
    std::ifstream in(path);
    std::string magic;
    int revision = 0;
    if (!(in >> magic >> revision >> out.threads >> out.lanes >> out.batch >> out.throughput_hps)) return false;
    if (magic != "YERBAS_CPU_POLICY" || revision != kCpuPolicyRevision ||
        out.threads == 0U || out.batch == 0U ||
        (out.lanes != 1U && out.lanes != 2U && out.lanes != 4U)) return false;
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
            << value.threads << ' ' << value.lanes << ' ' << value.batch << ' '
            << std::setprecision(12) << value.throughput_hps << '\n';
    } catch (...) {}
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

    if (mode == "off" || env_enabled("YERBAS_CPU_DISABLE_AUTOTUNE"))
        return TuneResult{ceiling, 1U, fallback_batch, 0.0, false, false};

    TuneResult cached{};
    const auto path = cache_path(hardware_threads, ceiling, mode);
    if (load_cache(path, cached) && cached.threads * cached.lanes <= ceiling) {
        std::cout << "[CPU tune] cached | mode=" << mode
                  << " | policy=" << cached.threads << 'x' << cached.lanes
                  << " | batch=" << cached.batch
                  << " | throughput=" << std::fixed << std::setprecision(2)
                  << cached.throughput_hps << " H/s" << std::defaultfloat << '\n';
        return cached;
    }

    const auto plans = plans_for(ceiling, fallback_batch, mode);
    std::cout << "[CPU tune] GhostRider production search"
              << " | mode=" << mode
              << " | hardware_threads=" << hardware_threads
              << " | ceiling=" << ceiling
              << " | candidates=" << plans.size()
              << " | schedules=3"
              << " | metric=median end-to-end H/s\n";

    TuneResult best{ceiling, 1U, fallback_batch, 0.0, false, false};
    for (const auto& plan : plans) {
        if (stopped(stop)) {
            best.interrupted = true;
            return best;
        }
        const double hps = benchmark_plan(plan, stop);
        if (env_enabled("YERBAS_DIAGNOSTICS")) {
            std::cout << "[CPU tune candidate] workers=" << plan.workers
                      << " | lanes=" << plan.lanes
                      << " | batch=" << plan.batch
                      << " | " << std::fixed << std::setprecision(2) << hps
                      << " H/s" << std::defaultfloat << '\n';
        }
        if (hps > best.throughput_hps) {
            best.threads = plan.workers;
            best.lanes = plan.lanes;
            best.batch = plan.batch;
            best.throughput_hps = hps;
        }
    }

    if (stopped(stop)) {
        best.interrupted = true;
        return best;
    }
    save_cache(path, best);
    std::cout << "[CPU tune] selected"
              << " | workers=" << best.threads
              << " | lanes=" << best.lanes
              << " | batch=" << best.batch
              << " | throughput=" << std::fixed << std::setprecision(2)
              << best.throughput_hps << " H/s"
              << " | cache=cpu-policy-v2" << std::defaultfloat << '\n';
    return best;
}

} // namespace yerbas::cpu
