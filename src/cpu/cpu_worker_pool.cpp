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
constexpr double kRotationWorkerRequiredGain = 1.03;
constexpr std::uint64_t kRotationWidthProbeSamples = 3U;
constexpr std::uint64_t kRotationWidthFinalSamples = 5U;
constexpr double kRotationWidthRequiredGain = 1.02;

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

std::string width_policy_name(const CnWidthPolicy& widths)
{
    std::ostringstream out;
    out << "rotation-width:";
    for (std::size_t i = 0; i < widths.size(); ++i) {
        if (i != 0) out << '/';
        out << widths[i];
    }
    return out.str();
}

std::size_t width_change_count(const CnWidthPolicy& widths,
                               const CnWidthPolicy& baseline) noexcept
{
    std::size_t changed = 0;
    for (std::size_t i = 0; i < widths.size(); ++i)
        if (widths[i] != baseline[i]) ++changed;
    return changed;
}

std::array<bool, 6> active_cn_variants(const ghostrider::StageSchedule& schedule)
{
    std::array<bool, 6> active{};
    for (const std::uint8_t stage : schedule) {
        if ((stage & ghostrider::kCryptoNightStageFlag) == 0) continue;
        const auto variant = static_cast<std::size_t>(stage & 0x7fU);
        if (variant < active.size()) active[variant] = true;
    }
    return active;
}

std::vector<CnWidthPolicy> rotation_width_candidates(const ghostrider::StageSchedule& schedule,
                                                     const CnWidthPolicy& baseline)
{
    std::vector<CnWidthPolicy> out;
    const auto add = [&](const CnWidthPolicy& widths) {
        if (std::find(out.begin(), out.end(), widths) == out.end()) out.push_back(widths);
    };
    add(baseline);
    const auto active = active_cn_variants(schedule);
    constexpr std::array<unsigned int, 3> widths{{1U, 2U, 4U}};
    for (std::size_t variant = 0; variant < active.size(); ++variant) {
        if (!active[variant]) continue;
        for (const unsigned int width : widths) {
            CnWidthPolicy candidate = baseline;
            candidate[variant] = width;
            add(candidate);
        }
    }
    return out;
}

bool width_plan_parity(const std::array<std::uint8_t, 80>& base_header,
                       const CnWidthPolicy& widths)
{
    std::array<std::array<std::uint8_t, 80>, 4> headers{};
    std::array<ghostrider::Work, 4> works{};
    std::array<ghostrider::Hash256, 4> optimized{};
    for (std::size_t i = 0; i < headers.size(); ++i) {
        headers[i] = base_header;
        write_nonce(headers[i], static_cast<std::uint32_t>(0xa5a50000U + i));
        works[i] = ghostrider::Work{headers[i].data(), headers[i].size()};
    }
    if (!ghostrider::hash_optimized_batch(works.data(), optimized.data(), works.size(), widths))
        return false;
    for (std::size_t i = 0; i < works.size(); ++i) {
        if (optimized[i] != ghostrider::hash_reference(works[i])) return false;
    }
    return true;
}

std::vector<unsigned int> rotation_worker_candidates(unsigned int configured_threads)
{
    std::vector<unsigned int> out;
    auto add = [&](unsigned int value) {
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

struct RotationWidthChoice {
    CnWidthPolicy widths{{1U, 1U, 1U, 1U, 1U, 1U}};
    bool probing{false};
    double selected_hps{0.0};
    double baseline_hps{0.0};
};

RotationWidthChoice choose_rotation_widths(std::uint64_t fingerprint,
                                           unsigned int workers,
                                           unsigned int lanes,
                                           unsigned int batch,
                                           const ghostrider::StageSchedule& schedule,
                                           const CnWidthPolicy& baseline,
                                           const std::vector<std::string>& rejected)
{
    const auto summaries = exact_fingerprint_policy_summaries(fingerprint);

    struct Measurement {
        CnWidthPolicy widths{{1U, 1U, 1U, 1U, 1U, 1U}};
        std::uint64_t samples{0U};
        double ewma_hps{0.0};
    };

    const auto measurement_for = [&](const CnWidthPolicy& candidate) {
        Measurement m{candidate, 0U, 0.0};
        const std::string policy = width_policy_name(candidate);
        for (const auto& summary : summaries) {
            if (summary.policy != policy || summary.threads != workers ||
                summary.lanes != lanes || summary.batch != batch)
                continue;
            m.samples = summary.samples;
            m.ewma_hps = summary.ewma_hps;
            break;
        }
        return m;
    };

    std::vector<Measurement> measured;
    const auto singles = rotation_width_candidates(schedule, baseline);
    measured.reserve(singles.size() + 4U);
    for (const auto& candidate : singles) {
        const std::string policy = width_policy_name(candidate);
        if (std::find(rejected.begin(), rejected.end(), policy) != rejected.end()) continue;
        measured.push_back(measurement_for(candidate));
    }
    if (measured.empty()) return {baseline, false, 0.0, 0.0};

    // Stage 1: cheaply screen the baseline and every one-variant alternative.
    auto single_probe = std::min_element(measured.begin(), measured.end(), [](const Measurement& a, const Measurement& b) {
        return a.samples < b.samples;
    });
    if (single_probe != measured.end() && single_probe->samples < kRotationWidthProbeSamples)
        return {single_probe->widths, true, single_probe->ewma_hps, 0.0};

    const Measurement* base = nullptr;
    for (const auto& m : measured)
        if (m.widths == baseline) { base = &m; break; }
    if (base == nullptr) return {baseline, false, 0.0, 0.0};

    // Keep only the individually proven winner for each active CN variant.
    // A variant must clear the same 2% whole-GhostRider gain gate before it is
    // allowed to participate in a combinational candidate.
    const auto active = active_cn_variants(schedule);
    std::array<unsigned int, 6> promising_width{};
    std::array<bool, 6> promising{};
    for (std::size_t variant = 0; variant < active.size(); ++variant) {
        if (!active[variant]) continue;
        const Measurement* best_variant = nullptr;
        for (const auto& m : measured) {
            if (width_change_count(m.widths, baseline) != 1U || m.widths[variant] == baseline[variant]) continue;
            bool this_variant_only = true;
            for (std::size_t i = 0; i < baseline.size(); ++i) {
                if (i != variant && m.widths[i] != baseline[i]) { this_variant_only = false; break; }
            }
            if (!this_variant_only) continue;
            if (best_variant == nullptr || m.ewma_hps > best_variant->ewma_hps) best_variant = &m;
        }
        if (best_variant != nullptr && best_variant->ewma_hps >= base->ewma_hps * kRotationWidthRequiredGain) {
            promising[variant] = true;
            promising_width[variant] = best_variant->widths[variant];
        }
    }

    // Stage 2: generate only combinations of individually proven winners.
    // GhostRider has three CN stages, so this normally means at most three
    // pair candidates plus one all-promising candidate rather than 27 tuples.
    std::vector<std::size_t> promising_variants;
    for (std::size_t i = 0; i < promising.size(); ++i)
        if (promising[i]) promising_variants.push_back(i);

    std::vector<CnWidthPolicy> combinations;
    const auto add_combination = [&](const CnWidthPolicy& widths) {
        if (width_change_count(widths, baseline) < 2U) return;
        if (std::find(combinations.begin(), combinations.end(), widths) == combinations.end())
            combinations.push_back(widths);
    };

    for (std::size_t a = 0; a < promising_variants.size(); ++a) {
        for (std::size_t b = a + 1; b < promising_variants.size(); ++b) {
            CnWidthPolicy candidate = baseline;
            candidate[promising_variants[a]] = promising_width[promising_variants[a]];
            candidate[promising_variants[b]] = promising_width[promising_variants[b]];
            add_combination(candidate);
        }
    }
    if (promising_variants.size() >= 3U) {
        CnWidthPolicy candidate = baseline;
        for (const auto variant : promising_variants)
            candidate[variant] = promising_width[variant];
        add_combination(candidate);
    }

    for (const auto& candidate : combinations) {
        const std::string policy = width_policy_name(candidate);
        if (std::find(rejected.begin(), rejected.end(), policy) != rejected.end()) continue;
        Measurement m = measurement_for(candidate);
        if (m.samples < kRotationWidthFinalSamples)
            return {m.widths, true, m.ewma_hps, base->ewma_hps};
        measured.push_back(m);
    }

    const Measurement* best = base;
    for (const auto& m : measured)
        if (m.ewma_hps > best->ewma_hps) best = &m;

    if (best->widths != baseline && best->ewma_hps < base->ewma_hps * kRotationWidthRequiredGain)
        best = base;
    return {best->widths, false, best->ewma_hps, base->ewma_hps};
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
          base_cn_widths(runtime_cn_widths()),
          active_widths(base_cn_widths),
          locked_widths(base_cn_widths),
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
        set_runtime_cn_widths(base_cn_widths);
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
        RotationWidthChoice width_choice{base_cn_widths, false, 0.0, 0.0};
        bool worker_plan_locked = false;
        bool width_plan_locked_now = false;
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

                if (active_fingerprint != locked_fingerprint) {
                    locked_fingerprint = active_fingerprint;
                    locked_workers = threads;
                    locked_selected_hps = 0.0;
                    locked_full_hps = 0.0;
                    rotation_plan_locked = false;
                    locked_widths = base_cn_widths;
                    active_widths = base_cn_widths;
                    locked_width_selected_hps = 0.0;
                    locked_width_baseline_hps = 0.0;
                    width_plan_locked = false;
                    rejected_width_policies.clear();
                    set_runtime_cn_widths(base_cn_widths);
                }
            }
            target_le = active_target;

            if (!tuning_measurement_mode()) {
                if (rotation_plan_locked && locked_fingerprint == active_fingerprint) {
                    worker_choice = {locked_workers, false, locked_selected_hps, locked_full_hps};
                    worker_plan_locked = true;
                } else {
                    worker_choice = choose_rotation_workers(active_fingerprint, threads, lanes, count);
                    if (!worker_choice.probing) {
                        locked_fingerprint = active_fingerprint;
                        locked_workers = std::max(1U, std::min(threads, worker_choice.workers));
                        locked_selected_hps = worker_choice.selected_hps;
                        locked_full_hps = worker_choice.full_hps;
                        rotation_plan_locked = true;
                        worker_plan_locked = true;
                        worker_choice.workers = locked_workers;
                    }
                }
            }
            active_workers = std::max(1U, std::min(threads, worker_choice.workers));

            active_widths = base_cn_widths;
            if (!tuning_measurement_mode() && worker_plan_locked) {
                if (width_plan_locked && locked_fingerprint == active_fingerprint) {
                    width_choice = {locked_widths, false, locked_width_selected_hps, locked_width_baseline_hps};
                    active_widths = locked_widths;
                    width_plan_locked_now = true;
                } else {
                    for (;;) {
                        width_choice = choose_rotation_widths(active_fingerprint, active_workers, lanes, count,
                                                              active_schedule, base_cn_widths,
                                                              rejected_width_policies);
                        const std::string policy = width_policy_name(width_choice.widths);
                        if (std::find(validated_width_policies.begin(), validated_width_policies.end(), policy) !=
                            validated_width_policies.end())
                            break;
                        if (width_plan_parity(header, width_choice.widths)) {
                            validated_width_policies.push_back(policy);
                            break;
                        }
                        rejected_width_policies.push_back(policy);
                        std::cout << "[CPU width] rotation=" << std::hex << active_fingerprint << std::dec
                                  << " | parity=FAIL | widths="
                                  << width_choice.widths[0] << '/' << width_choice.widths[1] << '/'
                                  << width_choice.widths[2] << '/' << width_choice.widths[3] << '/'
                                  << width_choice.widths[4] << '/' << width_choice.widths[5] << '\n';
                    }
                    active_widths = width_choice.widths;
                    if (!width_choice.probing) {
                        locked_widths = width_choice.widths;
                        locked_width_selected_hps = width_choice.selected_hps;
                        locked_width_baseline_hps = width_choice.baseline_hps;
                        width_plan_locked = true;
                        width_plan_locked_now = true;
                    }
                }
            }
            set_runtime_cn_widths(active_widths);

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
            FingerprintRuntimeSummary fp{};
            if (active_widths == base_cn_widths) {
                fp = record_fingerprint_runtime(active_fingerprint,
                                                active_cn_summary,
                                                "rotation",
                                                active_workers,
                                                lanes,
                                                count,
                                                hps);
            }

            FingerprintRuntimeSummary width_fp{};
            if (worker_plan_locked) {
                width_fp = record_fingerprint_runtime(active_fingerprint,
                                                      active_cn_summary,
                                                      width_policy_name(active_widths),
                                                      active_workers,
                                                      lanes,
                                                      count,
                                                      hps);
            }

            if ((run_count % 64U) == 0U) {
                auto sample_header = header;
                write_nonce(sample_header, start);
                sample_ghostrider_stages(sample_header, active_schedule, active_fingerprint);
            }

            const bool new_rotation = active_fingerprint != last_logged_fingerprint;
            const bool worker_plan_changed = active_workers != last_logged_active_workers;
            const bool width_plan_changed = active_widths != last_logged_widths;
            const bool periodic_diagnostic = diagnostics_enabled() && (run_count % 64U) == 0U;
            if (new_rotation || worker_plan_changed || width_plan_changed || periodic_diagnostic) {
                const auto learned = worker_plan_locked ? width_fp : fp;
                const char* width_kind = width_change_count(active_widths, base_cn_widths) > 1U ? "combined" : "single";
                std::cout << std::fixed << std::setprecision(2)
                          << "[CPU fingerprint] rotation=" << std::hex << std::setw(16) << std::setfill('0')
                          << active_fingerprint << std::dec << std::setfill(' ')
                          << " | CN=" << active_cn_summary
                          << " | policy=rotation workers=" << active_workers << '/' << threads
                          << " lanes=" << lanes << " batch=" << count
                          << " | phase=" << (worker_choice.probing ? "probe" : (worker_plan_locked ? "locked" : "selected"))
                          << " | width-phase=" << (!worker_plan_locked ? "waiting" : (width_choice.probing ? "probe" : (width_plan_locked_now ? "locked" : "selected")))
                          << " | width-kind=" << width_kind
                          << " | affinity=" << affinity_policy_name(affinity)
                          << " | widths=" << active_widths[0] << '/' << active_widths[1] << '/' << active_widths[2] << '/'
                          << active_widths[3] << '/' << active_widths[4] << '/' << active_widths[5]
                          << " | live=" << hps << " H/s"
                          << " | learned=" << learned.ewma_hps << " H/s"
                          << " | samples=" << learned.samples;
                if (!worker_choice.probing && worker_choice.full_hps > 0.0)
                    std::cout << " | full=" << worker_choice.full_hps << " H/s"
                              << " | selected=" << worker_choice.selected_hps << " H/s";
                if (worker_plan_locked && !width_choice.probing && width_choice.baseline_hps > 0.0)
                    std::cout << " | width-base=" << width_choice.baseline_hps << " H/s"
                              << " | width-selected=" << width_choice.selected_hps << " H/s";
                std::cout << std::defaultfloat << '\n';
                last_logged_fingerprint = active_fingerprint;
                last_logged_active_workers = active_workers;
                last_logged_widths = active_widths;
            }
        }

        std::vector<Candidate> merged;
        for (auto& result : results)
            merged.insert(merged.end(), result.begin(), result.end());
        return merged;
    }

    unsigned int threads;
    unsigned int lanes;
    AffinityPolicy affinity{AffinityPolicy::Unpinned};
    std::vector<unsigned int> affinity_cpus;
    CnWidthPolicy base_cn_widths{{1U, 1U, 1U, 1U, 1U, 1U}};
    CnWidthPolicy active_widths{{1U, 1U, 1U, 1U, 1U, 1U}};
    CnWidthPolicy locked_widths{{1U, 1U, 1U, 1U, 1U, 1U}};
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable work_ready;
    std::condition_variable all_done;
    bool stopping{false};
    std::uint64_t generation{0};
    unsigned int completed{0};
    std::array<std::uint8_t, 80> base_header{};
    std::array<std::uint8_t, 32> target_le{};
    std::uint32_t batch_start{0};
    unsigned int per_thread{0};
    const std::atomic_bool* stop_flag{nullptr};
    std::vector<std::vector<Candidate>> results;
    std::vector<std::uint32_t> assigned_starts;
    std::vector<unsigned int> assigned_counts;
    unsigned int active_workers{1U};
    std::uint64_t run_count{0};
    std::uint64_t last_logged_fingerprint{0};
    unsigned int last_logged_active_workers{0U};
    CnWidthPolicy last_logged_widths{{0U, 0U, 0U, 0U, 0U, 0U}};
    std::array<std::uint8_t, 76> active_header_key{};
    std::array<std::uint8_t, 32> active_target{};
    bool active_target_valid{false};
    ghostrider::StageSchedule active_schedule{};
    std::uint64_t active_fingerprint{0};
    std::string active_cn_summary;
    std::uint64_t locked_fingerprint{0};
    unsigned int locked_workers{1U};
    double locked_selected_hps{0.0};
    double locked_full_hps{0.0};
    bool rotation_plan_locked{false};
    double locked_width_selected_hps{0.0};
    double locked_width_baseline_hps{0.0};
    bool width_plan_locked{false};
    std::vector<std::string> validated_width_policies;
    std::vector<std::string> rejected_width_policies;
};

WorkerPool::WorkerPool(unsigned int threads, unsigned int lanes, AffinityPolicy affinity)
    : impl_(std::make_unique<Impl>(threads, lanes, affinity)) {}

WorkerPool::~WorkerPool() = default;
WorkerPool::WorkerPool(WorkerPool&&) noexcept = default;
WorkerPool& WorkerPool::operator=(WorkerPool&&) noexcept = default;

std::vector<Candidate> WorkerPool::run(const std::array<std::uint8_t, 80>& base_header,
                                       const std::array<std::uint8_t, 32>& target_le,
                                       std::uint32_t batch_start,
                                       unsigned int per_thread,
                                       const std::atomic_bool* stop_flag)
{
    return impl_->run(base_header, target_le, batch_start, per_thread, stop_flag);
}

} // namespace yerbas::cpu
