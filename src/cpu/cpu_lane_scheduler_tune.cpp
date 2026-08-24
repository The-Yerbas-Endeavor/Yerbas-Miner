#include "cpu/cpu_lane_scheduler_tune.h"
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
#include <sstream>
#include <thread>
#include <vector>

namespace yerbas::cpu {
namespace {

constexpr int kLaneTuneRevision = 1;
constexpr double kMinimumWin = 1.02;

bool env_enabled(const char* name)
{
    const char* v = std::getenv(name);
    return v != nullptr && *v != '\0' && std::string(v) != "0";
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

std::filesystem::path cache_path(unsigned int hardware_threads, unsigned int ceiling)
{
    std::ostringstream n;
    n << "cpu-lane-scheduler-rev" << kLaneTuneRevision
      << "-hw" << hardware_threads << "-max" << ceiling << ".txt";
    return cache_dir() / n.str();
}

void write_nonce(std::array<std::uint8_t,80>& h, std::uint32_t nonce)
{
    h[76] = static_cast<std::uint8_t>(nonce);
    h[77] = static_cast<std::uint8_t>(nonce >> 8);
    h[78] = static_cast<std::uint8_t>(nonce >> 16);
    h[79] = static_cast<std::uint8_t>(nonce >> 24);
}

std::array<std::array<std::uint8_t,80>,5> representative_headers()
{
    std::array<std::array<std::uint8_t,80>,5> headers{};
    for (std::size_t h = 0; h < headers.size(); ++h) {
        headers[h][0] = 4;
        for (std::size_t i = 4; i < 76; ++i)
            headers[h][i] = static_cast<std::uint8_t>((i * (19U + h * 4U) + 13U + h * 29U) & 0xffU);
    }
    return headers;
}

struct Plan { unsigned int workers; unsigned int lanes; };

std::vector<Plan> candidate_plans(unsigned int ceiling)
{
    std::set<std::pair<unsigned int,unsigned int>> uniq;
    auto add = [&](unsigned int workers, unsigned int lanes) {
        if (workers == 0 || lanes == 0) return;
        if (workers * lanes > ceiling) return;
        uniq.insert({workers, lanes});
    };

    add(ceiling, 1);
    if (ceiling > 1) add(ceiling - 1, 1);
    for (unsigned int lanes : {2U, 4U}) {
        const unsigned int max_workers = ceiling / lanes;
        if (max_workers == 0) continue;
        add(max_workers, lanes);
        if (max_workers > 1) add(max_workers - 1, lanes);
        add(1, lanes);
    }

    std::vector<Plan> out;
    for (const auto& p : uniq) out.push_back({p.first, p.second});
    std::sort(out.begin(), out.end(), [](const Plan& a, const Plan& b) {
        if (a.workers * a.lanes != b.workers * b.lanes)
            return a.workers * a.lanes > b.workers * b.lanes;
        return a.lanes < b.lanes;
    });
    return out;
}

double benchmark_plan(const Plan& plan)
{
    const auto headers = representative_headers();
    std::vector<double> samples;
    samples.reserve(headers.size());

    for (std::size_t sample = 0; sample < headers.size(); ++sample) {
        const auto start = std::chrono::steady_clock::now();
        std::vector<std::thread> workers;
        workers.reserve(plan.workers);
        for (unsigned int w = 0; w < plan.workers; ++w) {
            workers.emplace_back([&, w] {
                std::array<std::array<std::uint8_t,80>,4> lane_headers{};
                std::array<ghostrider::Work,4> works{};
                std::array<ghostrider::Hash256,4> hashes{};
                std::array<unsigned int,6> widths{{plan.lanes, plan.lanes, plan.lanes,
                                                  plan.lanes, plan.lanes, plan.lanes}};
                for (unsigned int lane = 0; lane < plan.lanes; ++lane) {
                    lane_headers[lane] = headers[sample];
                    write_nonce(lane_headers[lane], 0x43000000U +
                                static_cast<std::uint32_t>(sample * 0x1000U + w * 16U + lane));
                    works[lane] = {lane_headers[lane].data(), lane_headers[lane].size()};
                }
                if (plan.lanes == 1) {
                    hashes[0] = ghostrider::hash_optimized(works[0]);
                } else if (!ghostrider::hash_optimized_batch(works.data(), hashes.data(), plan.lanes, widths)) {
                    for (unsigned int lane = 0; lane < plan.lanes; ++lane)
                        hashes[lane] = ghostrider::hash_optimized(works[lane]);
                }
            });
        }
        for (auto& t : workers) t.join();
        const auto end = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(end - start).count();
        const double hashes = static_cast<double>(plan.workers * plan.lanes);
        if (seconds > 0.0) samples.push_back(hashes / seconds);
    }

    std::sort(samples.begin(), samples.end());
    return samples.empty() ? 0.0 : samples[samples.size()/2U];
}

bool load_cache(const std::filesystem::path& path, LaneSchedulerTuneResult& out)
{
    if (env_enabled("YERBAS_CPU_RETUNE") || env_enabled("YERBAS_CPU_LANE_RETUNE")) return false;
    std::ifstream in(path);
    std::string magic;
    int rev = 0;
    if (!(in >> magic >> rev >> out.workers >> out.lanes >> out.throughput_hps)) return false;
    if (magic != "YERBAS_CPU_LANE" || rev != kLaneTuneRevision) return false;
    if (out.workers == 0 || (out.lanes != 1 && out.lanes != 2 && out.lanes != 4)) return false;
    out.from_cache = true;
    return true;
}

void save_cache(const std::filesystem::path& path, const LaneSchedulerTuneResult& r)
{
    try {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream out(path, std::ios::trunc);
        if (!out) return;
        out << "YERBAS_CPU_LANE " << kLaneTuneRevision << ' '
            << r.workers << ' ' << r.lanes << ' ' << std::setprecision(12)
            << r.throughput_hps << '\n';
    } catch (...) {}
}

} // namespace

LaneSchedulerTuneResult tune_lane_scheduler(unsigned int hardware_threads,
                                            unsigned int configured_threads,
                                            const std::string& mode)
{
    hardware_threads = std::max(1U, hardware_threads);
    const unsigned int ceiling = configured_threads == 0U
        ? hardware_threads
        : std::max(1U, std::min(configured_threads, hardware_threads));

    LaneSchedulerTuneResult result{ceiling, 1U, 0.0, false};
    if (mode == "off" || mode == "simple") return result;

    const auto path = cache_path(hardware_threads, ceiling);
    if (load_cache(path, result)) {
        std::cout << "CPU lane scheduler: workers=" << result.workers
                  << " x lanes=" << result.lanes
                  << " | throughput=" << std::fixed << std::setprecision(2)
                  << result.throughput_hps << " H/s | source=cache | production=1way\n"
                  << std::defaultfloat;
        return result;
    }

    const auto plans = candidate_plans(ceiling);
    std::cout << "[CPU lane tune] complete GhostRider scheduler search"
              << " | ceiling=" << ceiling
              << " | plans=" << plans.size()
              << " | production remains 1-way until qualified\n";

    LaneSchedulerTuneResult best{ceiling, 1U, 0.0, false};
    double baseline = 0.0;
    for (const auto& plan : plans) {
        const double hps = benchmark_plan(plan);
        std::cout << "[CPU lane plan] workers=" << plan.workers
                  << " x lanes=" << plan.lanes
                  << " | concurrency=" << (plan.workers * plan.lanes)
                  << " | " << std::fixed << std::setprecision(2) << hps << " H/s\n"
                  << std::defaultfloat;
        if (plan.lanes == 1 && plan.workers == ceiling) baseline = hps;
        if (hps > best.throughput_hps) {
            best.workers = plan.workers;
            best.lanes = plan.lanes;
            best.throughput_hps = hps;
        }
    }

    if (best.lanes != 1 && baseline > 0.0 && best.throughput_hps < baseline * kMinimumWin) {
        best = {ceiling, 1U, baseline, false};
    }
    save_cache(path, best);

    std::cout << "[CPU lane tune] recommended | workers=" << best.workers
              << " x lanes=" << best.lanes
              << " | throughput=" << std::fixed << std::setprecision(2)
              << best.throughput_hps << " H/s"
              << " | production=1way | min-win=2%\n"
              << std::defaultfloat;
    return best;
}

} // namespace yerbas::cpu
