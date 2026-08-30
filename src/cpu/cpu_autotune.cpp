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

constexpr int kCpuPolicyRevision = 5;
constexpr double kCnLocalRequiredGain = 1.03;
constexpr double kWholePlanWidthRequiredGain = 1.005;
constexpr double kAffinityRequiredGain = 1.02;
constexpr unsigned int kCnProbePasses = 5;
constexpr unsigned int kAffinityProbePasses = 5;

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

double median(std::vector<double> samples)
{
    if (samples.empty()) return 0.0;
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2U];
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
    return median(std::move(samples));
}

double benchmark_plan_repeated(const Plan& plan,
                               const CnWidthPolicy& widths,
                               unsigned int passes,
                               const std::atomic_bool* stop)
{
    std::vector<double> samples;
    samples.reserve(passes);
    for (unsigned int pass = 0; pass < passes; ++pass) {
        if (stopped(stop)) return 0.0;
        const double hps = benchmark_plan(plan, widths, stop);
        if (hps > 0.0) samples.push_back(hps);
    }
    return median(std::move(samples));
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

struct CnProbeResult {
    bool parity{false};
    double scalar_hps{0.0};
    double pair_hps{0.0};
};

CnProbeResult probe_cn_pair(std::uint8_t variant, const std::atomic_bool* stop)
{
    CnProbeResult out{};
    ghostrider::Hash512 input0{};
    ghostrider::Hash512 input1{};
    for (std::size_t i = 0; i < input0.size(); ++i) {
        input0[i] = static_cast<std::uint8_t>((i * 29U + variant * 47U + 11U) & 0xffU);
        input1[i] = static_cast<std::uint8_t>((i * 53U + variant * 31U + 97U) & 0xffU);
    }

    ghostrider::Hash512 scalar0{}, scalar1{}, pair0{}, pair1{};
    if (!ghostrider::optimized_cn_stage(input0, variant, scalar0) ||
        !ghostrider::optimized_cn_stage(input1, variant, scalar1) ||
        !ghostrider::optimized_cn_pair_stage(input0, input1, variant, pair0, pair1))
        return out;
    out.parity = scalar0 == pair0 && scalar1 == pair1;
    if (!out.parity) return out;

    std::vector<double> scalar_samples;
    std::vector<double> pair_samples;
    scalar_samples.reserve(kCnProbePasses);
    pair_samples.reserve(kCnProbePasses);
    for (unsigned int pass = 0; pass < kCnProbePasses; ++pass) {
        if (stopped(stop)) return out;
        ghostrider::Hash512 a{}, b{};
        auto begin = std::chrono::steady_clock::now();
        (void)ghostrider::optimized_cn_stage(input0, variant, a);
        (void)ghostrider::optimized_cn_stage(input1, variant, b);
        auto end = std::chrono::steady_clock::now();
        const double scalar_seconds = std::chrono::duration<double>(end - begin).count();
        if (scalar_seconds > 0.0) scalar_samples.push_back(2.0 / scalar_seconds);

        begin = std::chrono::steady_clock::now();
        (void)ghostrider::optimized_cn_pair_stage(input0, input1, variant, a, b);
        end = std::chrono::steady_clock::now();
        const double pair_seconds = std::chrono::duration<double>(end - begin).count();
        if (pair_seconds > 0.0) pair_samples.push_back(2.0 / pair_seconds);
    }
    out.scalar_hps = median(std::move(scalar_samples));
    out.pair_hps = median(std::move(pair_samples));
    return out;
}

bool valid_width(unsigned int width)
{
    return width == 1U || width == 2U;
}

bool valid_affinity(unsigned int value)
{
    return value <= static_cast<unsigned int>(AffinityPolicy::PhysicalFirst);
}

bool load_cache(const std::filesystem::path& path, TuneResult& out)
{
    if (env_enabled("YERBAS_CPU_RETUNE")) return false;
    std::ifstream in(path);
    std::string magic;
    int revision = 0;
    unsigned int affinity = 0;
    if (!(in >> magic >> revision >> out.threads >> out.lanes >> out.batch)) return false;
    for (auto& width : out.cn_widths)
        if (!(in >> width)) return false;
    if (!(in >> affinity >> out.throughput_hps)) return false;
    if (magic != "YERBAS_CPU_POLICY" || revision != kCpuPolicyRevision ||
        out.threads == 0U || out.batch == 0U || !valid_width(out.lanes) || !valid_affinity(affinity)) return false;
    for (const auto width : out.cn_widths)
        if (!valid_width(width)) return false;
    if (out.lanes != max_width(out.cn_widths)) return false;
    out.affinity = static_cast<AffinityPolicy>(affinity);
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
        out << ' ' << static_cast<unsigned int>(value.affinity)
            << ' ' << std::setprecision(12) << value.throughput_hps << '\n';
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
    for (unsigned int candidate_ceiling = ceiling; candidate_ceiling <= hardware_threads; ++candidate_ceiling) {
        TuneResult candidate{};
        if (!load_cache(cache_path(hardware_threads, candidate_ceiling, mode), candidate)) continue;
        if (candidate.threads > ceiling || !parity_width_policy(candidate.cn_widths)) continue;
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

    set_runtime_affinity_policy(AffinityPolicy::Unpinned);
    if (mode == "off" || env_enabled("YERBAS_CPU_DISABLE_AUTOTUNE"))
        return TuneResult{ceiling, 1U, fallback_batch, scalar_widths, AffinityPolicy::Unpinned, 0.0, false, false};

    TuneResult cached{};
    const auto path = cache_path(hardware_threads, ceiling, mode);
    unsigned int source_ceiling = ceiling;
    if (load_compatible_cache(hardware_threads, ceiling, mode, cached, source_ceiling)) {
        set_runtime_cn_widths(cached.cn_widths);
        set_runtime_affinity_policy(cached.affinity);
        if (source_ceiling != ceiling) save_cache(path, cached);
        std::cout << "[CPU tune] cached | mode=" << mode
                  << " | workers=" << cached.threads
                  << " | batch=" << cached.batch
                  << " | CN widths=" << cached.cn_widths[0] << '/' << cached.cn_widths[1] << '/'
                  << cached.cn_widths[2] << '/' << cached.cn_widths[3] << '/'
                  << cached.cn_widths[4] << '/' << cached.cn_widths[5]
                  << " | affinity=" << affinity_policy_name(cached.affinity)
                  << " | throughput=" << std::fixed << std::setprecision(2) << cached.throughput_hps << " H/s"
                  << " | source=max" << source_ceiling
                  << (source_ceiling == ceiling ? "" : " (compatible)")
                  << std::defaultfloat << '\n';
        return cached;
    }

    const auto topology = detect_cpu_topology();
    std::cout << "[CPU topology] logical=" << topology.logical_cpus
              << " | physical=" << topology.physical_cores
              << " | detected=" << (topology.available ? "yes" : "no") << '\n';

    const auto plans = plans_for(ceiling, fallback_batch, mode);
    std::cout << "[CPU tune] GhostRider production search"
              << " | mode=" << mode
              << " | hardware_threads=" << hardware_threads
              << " | ceiling=" << ceiling
              << " | baseline_candidates=" << plans.size()
              << " | CN-probes=6x(scalar/2way)x5"
              << " | metric=end-to-end H/s\n";

    TuneResult best{ceiling, 1U, fallback_batch, scalar_widths, AffinityPolicy::Unpinned, 0.0, false, false};
    std::size_t plan_index = 0;
    for (const auto& plan : plans) {
        if (stopped(stop)) { best.interrupted = true; return best; }
        ++plan_index;
        std::cout << "[CPU tune] baseline " << plan_index << '/' << plans.size()
                  << " | workers=" << plan.workers << " | batch=" << plan.batch << " | testing..." << std::flush;
        set_runtime_affinity_policy(AffinityPolicy::Unpinned);
        const double hps = benchmark_plan(plan, scalar_widths, stop);
        if (hps > best.throughput_hps) {
            best.threads = plan.workers;
            best.batch = plan.batch;
            best.throughput_hps = hps;
        }
        std::cout << " done | " << std::fixed << std::setprecision(2) << hps
                  << " H/s | best=" << best.throughput_hps << " H/s" << std::defaultfloat << '\n';
    }

    if (stopped(stop)) { best.interrupted = true; return best; }

    const Plan selected_plan{best.threads, best.batch};
    CnWidthPolicy widths = scalar_widths;
    double whole_plan_hps = best.throughput_hps;

    for (std::size_t variant = 0; variant < widths.size(); ++variant) {
        if (stopped(stop)) { best.interrupted = true; return best; }
        const char* name = ghostrider::cryptonight_name(static_cast<std::uint8_t>(variant));
        const auto probe = probe_cn_pair(static_cast<std::uint8_t>(variant), stop);
        const double local_gain = probe.scalar_hps > 0.0 ? ((probe.pair_hps / probe.scalar_hps) - 1.0) * 100.0 : 0.0;
        const bool local_candidate = probe.parity && probe.pair_hps > probe.scalar_hps * kCnLocalRequiredGain;
        std::cout << "[CPU CN probe] " << name
                  << " | scalar=" << std::fixed << std::setprecision(2) << probe.scalar_hps
                  << " H/s | 2way=" << probe.pair_hps
                  << " H/s | gain=" << (local_gain >= 0.0 ? "+" : "") << local_gain << "%"
                  << " | parity=" << (probe.parity ? "PASS" : "FAIL")
                  << " | " << (local_candidate ? "candidate" : "scalar")
                  << std::defaultfloat << '\n';
        if (!local_candidate) continue;

        CnWidthPolicy candidate = widths;
        candidate[variant] = 2U;
        set_runtime_affinity_policy(AffinityPolicy::Unpinned);
        std::cout << "[CPU CN confirm] " << name << " | whole-GhostRider x3..." << std::flush;
        const double candidate_hps = benchmark_plan_repeated(selected_plan, candidate, 3U, stop);
        const bool parity_ok = parity_width_policy(candidate);
        const double gain_pct = whole_plan_hps > 0.0 ? ((candidate_hps / whole_plan_hps) - 1.0) * 100.0 : 0.0;
        const bool promote = parity_ok && candidate_hps > whole_plan_hps * kWholePlanWidthRequiredGain;
        std::cout << " done | " << std::fixed << std::setprecision(2) << candidate_hps
                  << " H/s | gain=" << (gain_pct >= 0.0 ? "+" : "") << gain_pct << "%"
                  << " | parity=" << (parity_ok ? "PASS" : "FAIL")
                  << " | " << (promote ? "promote" : "reject") << std::defaultfloat << '\n';
        if (promote) {
            widths = candidate;
            whole_plan_hps = candidate_hps;
        }
    }

    best.cn_widths = widths;
    best.lanes = max_width(widths);
    best.throughput_hps = whole_plan_hps;

#if defined(__linux__)
    if (topology.available && selected_plan.workers <= topology.logical_cpus) {
        std::cout << "[CPU affinity] A/B unpinned vs physical-first x" << kAffinityProbePasses << "...\n";
        set_runtime_affinity_policy(AffinityPolicy::Unpinned);
        const double unpinned_hps = benchmark_plan_repeated(selected_plan, widths, kAffinityProbePasses, stop);
        set_runtime_affinity_policy(AffinityPolicy::PhysicalFirst);
        const double pinned_hps = benchmark_plan_repeated(selected_plan, widths, kAffinityProbePasses, stop);
        const double gain_pct = unpinned_hps > 0.0 ? ((pinned_hps / unpinned_hps) - 1.0) * 100.0 : 0.0;
        const bool promote = pinned_hps > unpinned_hps * kAffinityRequiredGain;
        best.affinity = promote ? AffinityPolicy::PhysicalFirst : AffinityPolicy::Unpinned;
        best.throughput_hps = promote ? pinned_hps : unpinned_hps;
        std::cout << "[CPU affinity] unpinned=" << std::fixed << std::setprecision(2) << unpinned_hps
                  << " H/s | physical-first=" << pinned_hps
                  << " H/s | gain=" << (gain_pct >= 0.0 ? "+" : "") << gain_pct << "%"
                  << " | selected=" << affinity_policy_name(best.affinity) << std::defaultfloat << '\n';
    }
#endif

    set_runtime_cn_widths(best.cn_widths);
    set_runtime_affinity_policy(best.affinity);
    const bool parity_ok = parity_width_policy(best.cn_widths);
    std::cout << "[CPU tune] final validation | end-to-end + parity..." << std::flush;
    const double tuned_hps = benchmark_plan_repeated(selected_plan, best.cn_widths, 3U, stop);
    std::cout << " done | throughput=" << std::fixed << std::setprecision(2) << tuned_hps
              << " H/s | parity=" << (parity_ok ? "PASS" : "FAIL") << std::defaultfloat << '\n';

    if (!parity_ok) {
        best.cn_widths = scalar_widths;
        best.lanes = 1U;
        best.affinity = AffinityPolicy::Unpinned;
        set_runtime_cn_widths(best.cn_widths);
        set_runtime_affinity_policy(best.affinity);
    } else if (tuned_hps > 0.0) {
        best.throughput_hps = tuned_hps;
    }

    save_cache(path, best);
    std::cout << "[CPU tune] selected"
              << " | workers=" << best.threads
              << " | batch=" << best.batch
              << " | CN widths=" << best.cn_widths[0] << '/' << best.cn_widths[1] << '/'
              << best.cn_widths[2] << '/' << best.cn_widths[3] << '/'
              << best.cn_widths[4] << '/' << best.cn_widths[5]
              << " | affinity=" << affinity_policy_name(best.affinity)
              << " | parity=" << (parity_ok ? "PASS" : "FAIL-scalar")
              << " | throughput=" << std::fixed << std::setprecision(2) << best.throughput_hps << " H/s"
              << " | cache=cpu-policy-v5" << std::defaultfloat << '\n';
    return best;
}

} // namespace yerbas::cpu
