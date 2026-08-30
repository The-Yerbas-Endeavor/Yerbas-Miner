#include "cpu/cpu_worker_pool.h"

#include "cpu/fingerprint_runtime.h"
#include "cpu/stage_profiler.h"
#include "ghostrider/ghostrider.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

namespace yerbas::cpu {
namespace {

std::atomic_uint g_runtime_lane_width{1U};
std::array<std::atomic_uint, 6> g_runtime_cn_widths{{1U, 1U, 1U, 1U, 1U, 1U}};
std::atomic_bool g_tuning_measurement_mode{false};

constexpr std::uint64_t kRotationWorkerProbeSamples = 3U;
constexpr double kRotationWorkerRequiredGain = 1.01;

unsigned int normalize_lane_width(unsigned int lane_width) noexcept
{
    return lane_width == 2U || lane_width == 4U ? lane_width : 1U;
}

bool diagnostics_enabled()
{
    const char* value = std::getenv("YERBAS_DIAGNOSTICS");
    return value != nullptr && *value != '\0' && std::string(value) != "0";
}

void write_nonce(std::array<std::uint8_t, 80>& header, std::uint32_t nonce)
{
    header[76] = static_cast<std::uint8_t>(nonce);
    header[77] = static_cast<std::uint8_t>(nonce >> 8);
    header[78] = static_cast<std::uint8_t>(nonce >> 16);
    header[79] = static_cast<std::uint8_t>(nonce >> 24);
}

bool hash_meets_target(const ghostrider::Hash256& hash,
                       const std::array<std::uint8_t, 32>& target_le)
{
    for (int i = 31; i >= 0; --i) {
        const auto index = static_cast<std::size_t>(i);
        if (hash[index] < target_le[index]) return true;
        if (hash[index] > target_le[index]) return false;
    }
    return true;
}

std::string cn_summary(const ghostrider::StageSchedule& schedule)
{
    std::ostringstream out;
    bool first = true;
    for (const std::uint8_t stage : schedule) {
        if ((stage & ghostrider::kCryptoNightStageFlag) == 0) continue;
        const auto variant = static_cast<std::uint8_t>(stage & 0x7fU);
        if (!first) out << '/';
        out << ghostrider::cryptonight_name(variant);
        first = false;
    }
    return out.str();
}

std::vector<unsigned int> rotation_worker_candidates(unsigned int configured_threads)
{
    std::vector<unsigned int> out;
    const auto add = [&](unsigned int value) mutable {
        value = std::max(1U, std::min(configured_threads, value));
        if (std::find(out.begin(), out.end(), value) == out.end()) out.push_back(value);
    };

    add(configured_threads);
    static const CpuTopology topology = detect_cpu_topology();
    if (topology.physical_cores > 0U) add(topology.physical_cores);
    if (configured_threads > 1U) add(configured_threads - 1U);
    add(std::max(1U, configured_threads / 2U));
    return out;
}

struct RotationWorkerChoice {
    unsigned int workers{1U};
    bool probing{false};
    double selected_hps{0.0};
    double full_hps{0.0};
};

RotationWorkerChoice choose_rotation_workers(std::uint64_t fingerprint,
                                             unsigned int configured_threads,
                                             unsigned int lanes,
                                             unsigned int batch)
{
    const auto candidates = rotation_worker_candidates(configured_threads);
    const auto summaries = fingerprint_policy_summaries(fingerprint);

    struct Measurement {
        unsigned int workers{1U};
        std::uint64_t samples{0U};
        double ewma_hps{0.0};
    };
    std::vector<Measurement> measured;
    measured.reserve(candidates.size());

    for (const unsigned int candidate : candidates) {
        Measurement m{candidate, 0U, 0.0};
        for (const auto& summary : summaries) {
            if (summary.policy != "rotation" || summary.threads != candidate ||
                summary.lanes != lanes || summary.batch != batch)
                continue;
            if (summary.samples > m.samples) {
                m.samples = summary.samples;
                m.ewma_hps = summary.ewma_hps;
            }
        }
        measured.push_back(m);
    }

    auto probe = std::min_element(measured.begin(), measured.end(), [](const Measurement& a, const Measurement& b) {
        if (a.samples != b.samples) return a.samples < b.samples;
        return a.workers > b.workers;
    });
    if (probe != measured.end() && probe->samples < kRotationWorkerProbeSamples)
        return {probe->workers, true, probe->ewma_hps, 0.0};

    const Measurement* full = nullptr;
    const Measurement* best = nullptr;
    for (const auto& m : measured) {
        if (m.workers == configured_threads) full = &m;
        if (best == nullptr || m.ewma_hps > best->ewma_hps) best = &m;
    }
    if (full == nullptr || best == nullptr) return {configured_threads, false, 0.0, 0.0};

    if (best->workers != configured_threads && best->ewma_hps < full->ewma_hps * kRotationWorkerRequiredGain)
        best = full;
    return {best->workers, false, best->ewma_hps, full->ewma_hps};
}

} // namespace

void set_runtime_lane_width(unsigned int lane_width) noexcept
{
    g_runtime_lane_width.store(normalize_lane_width(lane_width), std::memory_order_relaxed);
}

unsigned int runtime_lane_width() noexcept
{
    return g_runtime_lane_width.load(std::memory_order_relaxed);
}

void set_runtime_cn_widths(const CnWidthPolicy& widths) noexcept
{
    for (std::size_t i = 0; i < widths.size(); ++i)
        g_runtime_cn_widths[i].store(normalize_lane_width(widths[i]), std::memory_order_relaxed);
}

CnWidthPolicy runtime_cn_widths() noexcept
{
    CnWidthPolicy out{};
    for (std::size_t i = 0; i < out.size(); ++i)
        out[i] = g_runtime_cn_widths[i].load(std::memory_order_relaxed);
    return out;
}

void set_tuning_measurement_mode(bool enabled) noexcept
{
    g_tuning_measurement_mode.store(enabled, std::memory_order_relaxed);
}

bool tuning_measurement_mode() noexcept
{
    return g_tuning_measurement_mode.load(std::memory_order_relaxed);
}

struct WorkerPool::Impl {
    explicit Impl(unsigned int requested_threads,
                  unsigned int requested_lanes,
                  AffinityPolicy requested_affinity)
        : threads(std::max(1u, requested_threads)),
          lanes(requested_lanes == 0U ? runtime_lane_width() : normalize_lane_width(requested_lanes)),
          affinity(requested_affinity),
          affinity_cpus(affinity_cpu_order(affinity, threads)),
          results(threads),
          assigned_starts(threads),
          assigned_counts(threads)
    {
        workers.reserve(threads);
        for (unsigned int index = 0; index < threads; ++index)
            workers.emplace_back([this, index] { worker_loop(index); });
    }

    ~Impl()
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
            ++generation;
        }
        work_ready.notify_all();
        for (auto& worker : workers)
            if (worker.joinable()) worker.join();
        if (!tuning_measurement_mode()) {
            flush_fingerprint_runtime();
            flush_stage_profiles();
        }
    }

    void worker_loop(unsigned int index)
    {
        if (index < affinity_cpus.size())
            (void)pin_current_thread_to_cpu(affinity_cpus[index]);

        std::uint64_t seen_generation = 0;
        for (;;) {
            std::array<std::uint8_t, 80> header{};
            std::array<std::uint8_t, 32> target{};
            std::uint32_t start = 0;
            unsigned int count = 0;
            const std::atomic_bool* stop = nullptr;

            {
                std::unique_lock<std::mutex> lock(mutex);
                work_ready.wait(lock, [&] { return stopping || generation != seen_generation; });
                if (stopping) return;
                seen_generation = generation;
                header = base_header;
                target = target_le;
                start = assigned_starts[index];
                count = assigned_counts[index];
                stop = stop_flag;
            }

            std::vector<Candidate> found;
            found.reserve(2);
            const auto widths = runtime_cn_widths();

            for (unsigned int i = 0; i < count;) {
                if (stop != nullptr && stop->load(std::memory_order_relaxed)) break;
                const unsigned int group = std::min(lanes, count - i);
                if (group == 1U || lanes == 1U) {
                    const std::uint32_t nonce = start + i;
                    auto lane_header = header;
                    write_nonce(lane_header, nonce);
                    const ghostrider::Work work{lane_header.data(), lane_header.size()};
                    const auto hash = ghostrider::hash_optimized(work);
                    if (hash_meets_target(hash, target)) found.push_back({nonce});
                    ++i;
                    continue;
                }

                std::array<std::array<std::uint8_t,80>,4> lane_headers{};
                std::array<ghostrider::Work,4> works{};
                std::array<ghostrider::Hash256,4> hashes{};
                std::array<std::uint32_t,4> nonces{};
                for (unsigned int lane = 0; lane < group; ++lane) {
                    lane_headers[lane] = header;
                    nonces[lane] = start + i + lane;
                    write_nonce(lane_headers[lane], nonces[lane]);
                    works[lane] = {lane_headers[lane].data(), lane_headers[lane].size()};
                }

                if (!ghostrider::hash_optimized_batch(works.data(), hashes.data(), group, widths)) {
                    for (unsigned int lane = 0; lane < group; ++lane)
                        hashes[lane] = ghostrider::hash_optimized(works[lane]);
                }
                for (unsigned int lane = 0; lane < group; ++lane) {
                    if (hash_meets_target(hashes[lane], target)) found.push_back({nonces[lane]});
                }
                i += group;
            }

            {
                std::lock_guard<std::mutex> lock(mutex);
                results[index] = std::move(found);
                ++completed;
                if (completed == threads) all_done.notify_one();
            }
        }
    }

    std::vector<Candidate> run(const std::array<std::uint8_t, 80>& header,
                               const std::array<std::uint8_t, 32>& target,
                               std::uint32_t start,
                               unsigned int count,
                               const std::atomic_bool* stop)
    {
        const auto perf_start = std::chrono::steady_clock::now();
        RotationWorkerChoice worker_choice{threads, false, 0.0, 0.0};
        {
            std::lock_guard<std::mutex> lock(mutex);
            base_header = header;

            std::array<std::uint8_t, 76> header_key{};
            std::copy_n(header.begin(), header_key.size(), header_key.begin());
            if (!active_target_valid || header_key != active_header_key) {
                active_header_key = header_key;
                active_target = target;
                active_target_valid = true;

                const ghostrider::Work work{header.data(), header.size()};
                active_schedule = ghostrider::stage_schedule_quiet(work);
                active_fingerprint = ghostrider::schedule_fingerprint(active_schedule);
                active_cn_summary = cn_summary(active_schedule);
            }
            target_le = active_target;

            if (!tuning_measurement_mode())
                worker_choice = choose_rotation_workers(active_fingerprint, threads, lanes, count);
            active_workers = std::max(1U, std::min(threads, worker_choice.workers));

            const std::uint64_t total_hashes = static_cast<std::uint64_t>(threads) * count;
            const std::uint64_t base = total_hashes / active_workers;
            const std::uint64_t remainder = total_hashes % active_workers;
            std::uint64_t offset = 0;
            for (unsigned int index = 0; index < threads; ++index) {
                if (index < active_workers) {
                    const std::uint64_t assigned = base + (index < remainder ? 1U : 0U);
                    assigned_starts[index] = static_cast<std::uint32_t>(static_cast<std::uint64_t>(start) + offset);
                    assigned_counts[index] = static_cast<unsigned int>(assigned);
                    offset += assigned;
                } else {
                    assigned_starts[index] = start;
                    assigned_counts[index] = 0U;
                }
            }

            batch_start = start;
            per_thread = count;
            stop_flag = stop;
            completed = 0;
            for (auto& result : results) result.clear();
            ++generation;
        }
        work_ready.notify_all();

        std::unique_lock<std::mutex> lock(mutex);
        all_done.wait(lock, [&] { return completed == threads; });
        lock.unlock();

        const auto perf_end = std::chrono::steady_clock::now();
        ++run_count;
        const double elapsed_ms = std::chrono::duration<double, std::milli>(perf_end - perf_start).count();
        const std::uint64_t hashes = static_cast<std::uint64_t>(count) * static_cast<std::uint64_t>(threads);
        const double hps = elapsed_ms > 0.0 ? static_cast<double>(hashes) * 1000.0 / elapsed_ms : 0.0;

        if (!tuning_measurement_mode() && !(stop != nullptr && stop->load(std::memory_order_relaxed))) {
            const auto fp = record_fingerprint_runtime(active_fingerprint,
                                                       active_cn_summary,
                                                       "rotation",
                                                       active_workers,
                                                       lanes,
                                                       count,
                                                       hps);

            if ((run_count % 64U) == 0U) {
                auto sample_header = header;
                write_nonce(sample_header, start);
                sample_ghostrider_stages(sample_header, active_schedule, active_fingerprint);
            }

            const bool new_rotation = active_fingerprint != last_logged_fingerprint;
            const bool worker_plan_changed = active_workers != last_logged_active_workers;
            const bool periodic_diagnostic = diagnostics_enabled() && (run_count % 64U) == 0U;
            if (new_rotation || worker_plan_changed || periodic_diagnostic) {
                const auto widths = runtime_cn_widths();
                std::cout << std::fixed << std::setprecision(2)
                          << "[CPU fingerprint] rotation=" << std::hex << std::setw(16) << std::setfill('0')
                          << active_fingerprint << std::dec << std::setfill(' ')
                          << " | CN=" << active_cn_summary
                          << " | policy=rotation workers=" << active_workers << '/' << threads
                          << " lanes=" << lanes << " batch=" << count
                          << " | phase=" << (worker_choice.probing ? "probe" : "selected")
                          << " | affinity=" << affinity_policy_name(affinity)
                          << " | widths=" << widths[0] << '/' << widths[1] << '/' << widths[2] << '/'
                          << widths[3] << '/' << widths[4] << '/' << widths[5]
                          << " | live=" << hps << " H/s"
                          << " | learned=" << fp.ewma_hps << " H/s"
                          << " | samples=" << fp.samples;
                if (!worker_choice.probing && worker_choice.full_hps > 0.0)
                    std::cout << " | full=" << worker_choice.full_hps << " H/s"
                              << " | selected=" << worker_choice.selected_hps << " H/s";
                std::cout << std::defaultfloat << '\n';
                last_logged_fingerprint = active_fingerprint;
                last_logged_active_workers = active_workers;
            }
        }

        std::size_t total_candidates = 0;
        for (const auto& result : results) total_candidates += result.size();
        std::vector<Candidate> combined;
        combined.reserve(total_candidates);
        for (auto& result : results)
            combined.insert(combined.end(), result.begin(), result.end());
        return combined;
    }

    const unsigned int threads;
    const unsigned int lanes;
    const AffinityPolicy affinity;
    const std::vector<unsigned int> affinity_cpus;
    std::vector<std::thread> workers;
    std::vector<std::vector<Candidate>> results;
    std::vector<std::uint32_t> assigned_starts;
    std::vector<unsigned int> assigned_counts;

    std::mutex mutex;
    std::condition_variable work_ready;
    std::condition_variable all_done;
    bool stopping{false};
    std::uint64_t generation{0};
    unsigned int completed{0};
    unsigned int active_workers{1U};
    std::uint64_t run_count{0};
    std::uint64_t last_logged_fingerprint{0};
    unsigned int last_logged_active_workers{0U};

    std::array<std::uint8_t, 80> base_header{};
    std::array<std::uint8_t, 32> target_le{};
    std::uint32_t batch_start{0};
    unsigned int per_thread{0};
    const std::atomic_bool* stop_flag{nullptr};

    std::array<std::uint8_t, 76> active_header_key{};
    std::array<std::uint8_t, 32> active_target{};
    bool active_target_valid{false};
    ghostrider::StageSchedule active_schedule{};
    std::uint64_t active_fingerprint{0};
    std::string active_cn_summary;
};

WorkerPool::WorkerPool(unsigned int thread_count,
                       unsigned int lane_width,
                       AffinityPolicy affinity)
    : impl_(std::make_unique<Impl>(thread_count, lane_width, affinity))
{
}

WorkerPool::~WorkerPool() = default;

unsigned int WorkerPool::thread_count() const noexcept
{
    return impl_->threads;
}

unsigned int WorkerPool::lane_width() const noexcept
{
    return impl_->lanes;
}

AffinityPolicy WorkerPool::affinity_policy() const noexcept
{
    return impl_->affinity;
}

std::vector<Candidate> WorkerPool::run(const std::array<std::uint8_t, 80>& base_header,
                                       const std::array<std::uint8_t, 32>& target_le,
                                       std::uint32_t batch_start,
                                       unsigned int per_thread,
                                       const std::atomic_bool* stop)
{
    return impl_->run(base_header, target_le, batch_start, per_thread, stop);
}

} // namespace yerbas::cpu
