#include "cpu/cpu_lane_scheduler_tune.h"
#include "ghostrider/ghostrider.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>
#include <vector>

namespace yerbas::cpu {
namespace {

constexpr int kLaneTuneRevision = 4;
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

std::filesystem::path cache_path(unsigned int hardware_threads,
                                 unsigned int ceiling,
                                 unsigned int batch)
{
    std::ostringstream n;
    n << "cpu-lane-scheduler-rev" << kLaneTuneRevision
      << "-hw" << hardware_threads
      << "-max" << ceiling
      << "-batch" << batch << ".txt";
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

bool validate_plan(const Plan& plan)
{
    if (plan.lanes == 1) return true;
    auto header = representative_headers().front();
    std::array<std::array<std::uint8_t,80>,4> lane_headers{};
    std::array<ghostrider::Work,4> works{};
    std::array<ghostrider::Hash256,4> hashes{};
    std::array<unsigned int,6> widths{{plan.lanes, plan.lanes, plan.lanes,
                                      plan.lanes, plan.lanes, plan.lanes}};
    for (unsigned int lane = 0; lane < plan.lanes; ++lane) {
        lane_headers[lane] = header;
        write_nonce(lane_headers[lane], 0x42000000U + lane);
        works[lane] = {lane_headers[lane].data(), lane_headers[lane].size()};
    }
    if (!ghostrider::hash_optimized_batch(works.data(), hashes.data(), plan.lanes, widths)) return false;
    for (unsigned int lane = 0; lane < plan.lanes; ++lane) {
        if (hashes[lane] != ghostrider::hash_optimized(works[lane])) return false;
    }
    return true;
}

double benchmark_plan(const Plan& plan, unsigned int batch)
{
    if (!validate_plan(plan)) return 0.0;
    const auto headers = representative_headers();
    std::vector<double> samples;
    samples.reserve(headers.size());
    batch = std::max(1U, batch);

    for (std::size_t sample = 0; sample < headers.size(); ++sample) {
        const auto start = std::chrono::steady_clock::now();
        std::vector<std::thread> workers;
        workers.reserve(plan.workers);
        std::atomic_bool failed{false};

        for (unsigned int w = 0; w < plan.workers; ++w) {
            workers.emplace_back([&, w] {
                std::array<unsigned int,6> widths{{plan.lanes, plan.lanes, plan.lanes,
                                                  plan.lanes, plan.lanes, plan.lanes}};
                for (unsigned int done = 0; done < batch && !failed.load(std::memory_order_relaxed);) {
                    const unsigned int group = std::min(plan.lanes, batch - done);
                    std::array<std::array<std::uint8_t,80>,4> lane_headers{};
                    std::array<ghostrider::Work,4> works{};
                    std::array<ghostrider::Hash256,4> hashes{};
                    for (unsigned int lane = 0; lane < group; ++lane) {
                        lane_headers[lane] = headers[sample];
                        write_nonce(lane_headers[lane], 0x43000000U +
                                    static_cast<std::uint32_t>(sample * 0x100000U + w * batch + done + lane));
                        works[lane] = {lane_headers[lane].data(), lane_headers[lane].size()};
                    }
                    if (group == 1) {
                        hashes[0] = ghostrider::hash_optimized(works[0]);
                    } else if (!ghostrider::hash_optimized_batch(works.data(), hashes.data(), group, widths)) {
                        failed.store(true, std::memory_order_relaxed);
                        break;
                    }
                    done += group;
                }
            });
        }
        for (auto& t : workers) t.join();
        if (failed.load(std::memory_order_relaxed)) return 0.0;

        const auto end = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(end - start).count();
        const double hashes = static_cast<double>(plan.workers) * static_cast<double>(batch);
        if (seconds > 0.0) samples.push_back(hashes / seconds);
    }

    std::sort(samples.begin(), samples.end());
    return samples.empty() ? 0.0 : samples[samples.size()/2U];
}

struct PairGate {
    std::mutex mutex;
    std::condition_variable cv;
    std::array<ghostrider::Hash512,2> input{};
    std::array<ghostrider::Hash512,2> output{};
    std::uint8_t variant{0};
    unsigned int arrived{0};
    std::uint64_t epoch{0};
    bool ok{true};
};

bool cooperative_pair_stage(PairGate& gate,
                            unsigned int side,
                            const ghostrider::Hash512& input,
                            std::uint8_t variant,
                            ghostrider::Hash512& output,
                            const std::atomic_bool& failed)
{
    std::unique_lock<std::mutex> lock(gate.mutex);
    const std::uint64_t my_epoch = gate.epoch;
    gate.input[side] = input;
    if (gate.arrived == 0) gate.variant = variant;
    else if (gate.variant != variant) gate.ok = false;
    ++gate.arrived;

    if (gate.arrived == 2) {
        if (gate.ok && !failed.load(std::memory_order_relaxed)) {
            gate.ok = ghostrider::optimized_cn_pair_stage(gate.input[0], gate.input[1],
                                                          gate.variant,
                                                          gate.output[0], gate.output[1]);
        }
        gate.arrived = 0;
        ++gate.epoch;
        output = gate.output[side];
        const bool ok = gate.ok;
        gate.ok = true;
        lock.unlock();
        gate.cv.notify_all();
        return ok;
    }

    gate.cv.wait(lock, [&] {
        return gate.epoch != my_epoch || failed.load(std::memory_order_relaxed);
    });
    if (failed.load(std::memory_order_relaxed)) return false;
    output = gate.output[side];
    return true;
}

double benchmark_cooperative_pairs(unsigned int workers_count,
                                   unsigned int batch,
                                   bool& parity_ok)
{
    parity_ok = false;
    if (workers_count < 2) return 0.0;
    const auto headers = representative_headers();
    std::vector<double> samples;
    samples.reserve(headers.size());
    batch = std::max(1U, batch);

    for (std::size_t sample = 0; sample < headers.size(); ++sample) {
        const auto schedule = ghostrider::stage_schedule_quiet(
            ghostrider::Work{headers[sample].data(), headers[sample].size()});

        const unsigned int pair_count = workers_count / 2U;
        std::vector<std::unique_ptr<PairGate>> gates;
        gates.reserve(pair_count);
        for (unsigned int i = 0; i < pair_count; ++i)
            gates.emplace_back(std::make_unique<PairGate>());

        std::vector<ghostrider::Hash256> first_outputs(workers_count);
        std::vector<std::array<std::uint8_t,80>> first_headers(workers_count);
        std::atomic_bool failed{false};
        std::vector<std::thread> workers;
        workers.reserve(workers_count);

        const auto start = std::chrono::steady_clock::now();
        for (unsigned int w = 0; w < workers_count; ++w) {
            workers.emplace_back([&, w] {
                for (unsigned int done = 0; done < batch && !failed.load(std::memory_order_relaxed); ++done) {
                    auto header = headers[sample];
                    const std::uint32_t nonce = 0x45000000U +
                        static_cast<std::uint32_t>(sample * 0x100000U + w * batch + done);
                    write_nonce(header, nonce);
                    if (done == 0) first_headers[w] = header;

                    ghostrider::Hash512 state{};
                    for (std::size_t stage_index = 0;
                         stage_index < schedule.size() && !failed.load(std::memory_order_relaxed);
                         ++stage_index) {
                        const std::uint8_t stage = schedule[stage_index];
                        if ((stage & ghostrider::kCryptoNightStageFlag) == 0) {
                            ghostrider::Hash512 next{};
                            const ghostrider::Work input = stage_index == 0
                                ? ghostrider::Work{header.data(), header.size()}
                                : ghostrider::Work{state.data(), state.size()};
                            if (!ghostrider::optimized_core_stage(input, stage, next)) {
                                failed.store(true, std::memory_order_relaxed);
                                break;
                            }
                            state = next;
                            continue;
                        }

                        const std::uint8_t variant = static_cast<std::uint8_t>(stage & 0x7fU);
                        ghostrider::Hash512 next{};
                        if (w < pair_count * 2U) {
                            if (!cooperative_pair_stage(*gates[w / 2U], w & 1U,
                                                        state, variant, next, failed)) {
                                failed.store(true, std::memory_order_relaxed);
                                break;
                            }
                        } else if (!ghostrider::optimized_cn_stage(state, variant, next)) {
                            failed.store(true, std::memory_order_relaxed);
                            break;
                        }
                        state = next;
                    }

                    if (done == 0 && !failed.load(std::memory_order_relaxed))
                        std::memcpy(first_outputs[w].data(), state.data(), first_outputs[w].size());
                }
            });
        }
        for (auto& t : workers) t.join();
        if (failed.load(std::memory_order_relaxed)) return 0.0;
        const auto end = std::chrono::steady_clock::now();

        for (unsigned int w = 0; w < workers_count; ++w) {
            const ghostrider::Work ref_work{first_headers[w].data(), first_headers[w].size()};
            if (first_outputs[w] != ghostrider::hash_optimized(ref_work)) return 0.0;
        }

        const double seconds = std::chrono::duration<double>(end - start).count();
        const double hashes = static_cast<double>(workers_count) * static_cast<double>(batch);
        if (seconds > 0.0) samples.push_back(hashes / seconds);
    }

    std::sort(samples.begin(), samples.end());
    parity_ok = !samples.empty();
    return samples.empty() ? 0.0 : samples[samples.size()/2U];
}

bool load_cache(const std::filesystem::path& path, LaneSchedulerTuneResult& out)
{
    if (env_enabled("YERBAS_CPU_RETUNE") || env_enabled("YERBAS_CPU_LANE_RETUNE")) return false;
    std::ifstream in(path);
    std::string magic;
    int rev = 0;
    if (!(in >> magic >> rev >> out.workers >> out.lanes >> out.batch >> out.throughput_hps)) return false;
    if (magic != "YERBAS_CPU_LANE" || rev != kLaneTuneRevision) return false;
    if (out.workers == 0 || out.batch == 0 || (out.lanes != 1 && out.lanes != 2 && out.lanes != 4)) return false;
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
            << r.workers << ' ' << r.lanes << ' ' << r.batch << ' '
            << std::setprecision(12) << r.throughput_hps << '\n';
    } catch (...) {}
}

} // namespace

LaneSchedulerTuneResult tune_lane_scheduler(unsigned int hardware_threads,
                                            unsigned int configured_threads,
                                            unsigned int production_batch,
                                            const std::string& mode)
{
    hardware_threads = std::max(1U, hardware_threads);
    const unsigned int ceiling = configured_threads == 0U
        ? hardware_threads
        : std::max(1U, std::min(configured_threads, hardware_threads));
    const unsigned int batch = std::max(1U, production_batch);

    LaneSchedulerTuneResult result{ceiling, 1U, batch, 0.0, false};
    if (mode == "off" || mode == "simple") return result;

    const auto path = cache_path(hardware_threads, ceiling, batch);
    if (load_cache(path, result)) {
        std::cout << "CPU lane scheduler: workers=" << result.workers
                  << " x lanes=" << result.lanes
                  << " | batch=" << result.batch
                  << " | throughput=" << std::fixed << std::setprecision(2)
                  << result.throughput_hps << " H/s | source=cache\n"
                  << std::defaultfloat;
        return result;
    }

    const auto plans = candidate_plans(ceiling);
    std::cout << "[CPU lane tune] sustained GhostRider scheduler search"
              << " | ceiling=" << ceiling
              << " | batch=" << batch
              << " | plans=" << plans.size()
              << " | parity=required\n";

    LaneSchedulerTuneResult best{ceiling, 1U, batch, 0.0, false};
    double baseline = 0.0;
    for (const auto& plan : plans) {
        const double hps = benchmark_plan(plan, batch);
        std::cout << "[CPU lane plan] workers=" << plan.workers
                  << " x lanes=" << plan.lanes
                  << " | concurrency=" << (plan.workers * plan.lanes)
                  << " | batch=" << batch
                  << " | " << std::fixed << std::setprecision(2) << hps << " H/s\n"
                  << std::defaultfloat;
        if (plan.lanes == 1 && plan.workers == ceiling) baseline = hps;
        if (hps > best.throughput_hps) {
            best.workers = plan.workers;
            best.lanes = plan.lanes;
            best.throughput_hps = hps;
        }
    }

    bool cooperative_parity = false;
    const double cooperative_hps = benchmark_cooperative_pairs(ceiling, batch, cooperative_parity);
    std::cout << "[CPU cooperative plan] workers=" << ceiling
              << " | core=" << ceiling << "way"
              << " | CN=" << (ceiling / 2U) << "x2way"
              << " | batch=" << batch
              << " | " << std::fixed << std::setprecision(2) << cooperative_hps << " H/s"
              << " | parity=" << (cooperative_parity ? "PASS" : "FAIL");
    if (cooperative_parity && baseline > 0.0)
        std::cout << " | gain=" << std::setprecision(2)
                  << ((cooperative_hps / baseline - 1.0) * 100.0) << '%';
    std::cout << '\n' << std::defaultfloat;

    if (cooperative_parity && baseline > 0.0 && cooperative_hps >= baseline * kMinimumWin) {
        std::cout << "[CPU cooperative plan] QUALIFIED | exceeds 1-way baseline by >=2%"
                  << " | production remains gated until worker-pool integration\n";
    } else {
        std::cout << "[CPU cooperative plan] not qualified | production remains 1-way\n";
    }

    if (best.lanes != 1 && baseline > 0.0 && best.throughput_hps < baseline * kMinimumWin)
        best = {ceiling, 1U, batch, baseline, false};

    save_cache(path, best);
    std::cout << "[CPU lane tune] recommended | workers=" << best.workers
              << " x lanes=" << best.lanes
              << " | batch=" << best.batch
              << " | throughput=" << std::fixed << std::setprecision(2)
              << best.throughput_hps << " H/s"
              << " | min-win=2%\n"
              << std::defaultfloat;
    return best;
}

} // namespace yerbas::cpu
