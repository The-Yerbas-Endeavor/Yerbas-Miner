#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace yerbas::cpu {

struct FingerprintRuntimeSummary {
    std::uint64_t samples{0};
    double ewma_hps{0.0};
    double best_hps{0.0};
};

struct FingerprintPolicySummary {
    std::string policy;
    unsigned int threads{1};
    unsigned int lanes{1};
    unsigned int batch{1};
    std::uint64_t samples{0};
    double ewma_hps{0.0};
    double best_hps{0.0};
};

namespace fingerprint_detail {

inline std::filesystem::path cache_dir()
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

inline std::filesystem::path cache_path()
{
    return cache_dir() / "cpu-fingerprint-runtime-rev2.txt";
}

inline std::filesystem::path legacy_cache_path()
{
    return cache_dir() / "cpu-fingerprint-runtime-rev1.txt";
}

inline std::string key(std::uint64_t fingerprint,
                       const std::string& policy,
                       unsigned int threads,
                       unsigned int lanes,
                       unsigned int batch)
{
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << fingerprint
        << std::dec << ':' << policy << ':' << threads << ':' << lanes << ':' << batch;
    return out.str();
}

struct Entry {
    std::uint64_t fingerprint{0};
    std::string policy{"standard"};
    unsigned int threads{1};
    unsigned int lanes{1};
    unsigned int batch{1};
    std::uint64_t samples{0};
    double ewma_hps{0.0};
    double best_hps{0.0};
    std::string cn;
};

class Store {
public:
    Store() { load(); }

    FingerprintRuntimeSummary record(std::uint64_t fingerprint,
                                     const std::string& cn,
                                     const std::string& policy,
                                     unsigned int threads,
                                     unsigned int lanes,
                                     unsigned int batch,
                                     double hps)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto k = key(fingerprint, policy, threads, lanes, batch);
        auto& e = entries_[k];
        if (e.samples == 0) {
            e.fingerprint = fingerprint;
            e.policy = policy;
            e.threads = threads;
            e.lanes = lanes;
            e.batch = batch;
            e.cn = cn;
            e.ewma_hps = hps;
            e.best_hps = hps;
        } else {
            constexpr double alpha = 0.20;
            e.ewma_hps = e.ewma_hps * (1.0 - alpha) + hps * alpha;
            e.best_hps = std::max(e.best_hps, hps);
        }
        ++e.samples;
        if ((e.samples % 8U) == 0U) save_locked();
        return {e.samples, e.ewma_hps, e.best_hps};
    }

    std::vector<FingerprintPolicySummary> policies(std::uint64_t fingerprint) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::string cn;
        for (const auto& pair : entries_) {
            const auto& e = pair.second;
            if (e.fingerprint == fingerprint && !e.cn.empty()) {
                cn = e.cn;
                break;
            }
        }

        struct Aggregate {
            FingerprintPolicySummary summary;
            long double weighted_hps{0.0};
        };
        std::map<std::string, Aggregate> aggregates;

        for (const auto& pair : entries_) {
            const auto& e = pair.second;
            if (cn.empty()) {
                if (e.fingerprint != fingerprint) continue;
            } else if (e.cn != cn) {
                continue;
            }

            std::ostringstream aggregate_key;
            aggregate_key << e.policy << ':' << e.threads << ':' << e.lanes << ':' << e.batch;
            auto& a = aggregates[aggregate_key.str()];
            if (a.summary.samples == 0) {
                a.summary.policy = e.policy;
                a.summary.threads = e.threads;
                a.summary.lanes = e.lanes;
                a.summary.batch = e.batch;
            }
            a.summary.samples += e.samples;
            a.summary.best_hps = std::max(a.summary.best_hps, e.best_hps);
            a.weighted_hps += static_cast<long double>(e.ewma_hps) *
                              static_cast<long double>(e.samples);
        }

        std::vector<FingerprintPolicySummary> out;
        out.reserve(aggregates.size());
        for (auto& pair : aggregates) {
            auto& a = pair.second;
            if (a.summary.samples > 0) {
                a.summary.ewma_hps = static_cast<double>(
                    a.weighted_hps / static_cast<long double>(a.summary.samples));
            }
            out.push_back(a.summary);
        }
        return out;
    }

    std::vector<FingerprintPolicySummary> exact_policies(std::uint64_t fingerprint) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<FingerprintPolicySummary> out;
        for (const auto& pair : entries_) {
            const auto& e = pair.second;
            if (e.fingerprint != fingerprint) continue;
            out.push_back(FingerprintPolicySummary{e.policy, e.threads, e.lanes,
                                                   e.batch, e.samples,
                                                   e.ewma_hps, e.best_hps});
        }
        return out;
    }

    std::optional<FingerprintPolicySummary> recommended(std::uint64_t fingerprint,
                                                        std::uint64_t minimum_samples,
                                                        double minimum_gain) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const Entry* standard = nullptr;
        const Entry* best = nullptr;
        for (const auto& pair : entries_) {
            const auto& e = pair.second;
            if (e.fingerprint != fingerprint || e.samples < minimum_samples) continue;
            if (e.policy == "standard") {
                if (standard == nullptr || e.ewma_hps > standard->ewma_hps) standard = &e;
            }
            if (best == nullptr || e.ewma_hps > best->ewma_hps) best = &e;
        }
        if (standard == nullptr || best == nullptr) return std::nullopt;
        if (best->policy != "standard" && best->ewma_hps < standard->ewma_hps * minimum_gain)
            best = standard;
        return FingerprintPolicySummary{best->policy, best->threads, best->lanes,
                                        best->batch, best->samples,
                                        best->ewma_hps, best->best_hps};
    }

    void flush()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        save_locked();
    }

private:
    void load_v2()
    {
        std::ifstream in(cache_path());
        std::string magic;
        int rev = 0;
        if (!(in >> magic >> rev) || magic != "YERBAS_CPU_FP" || rev != 2) return;
        Entry e;
        while (in >> std::hex >> e.fingerprint >> std::dec
                  >> std::quoted(e.policy)
                  >> e.threads >> e.lanes >> e.batch >> e.samples
                  >> e.ewma_hps >> e.best_hps >> std::quoted(e.cn)) {
            entries_[key(e.fingerprint, e.policy, e.threads, e.lanes, e.batch)] = e;
        }
    }

    void import_legacy_v1()
    {
        if (!entries_.empty()) return;
        std::ifstream in(legacy_cache_path());
        std::string magic;
        int rev = 0;
        if (!(in >> magic >> rev) || magic != "YERBAS_CPU_FP" || rev != 1) return;
        Entry e;
        e.policy = "standard";
        while (in >> std::hex >> e.fingerprint >> std::dec
                  >> e.threads >> e.lanes >> e.batch >> e.samples
                  >> e.ewma_hps >> e.best_hps >> std::quoted(e.cn)) {
            entries_[key(e.fingerprint, e.policy, e.threads, e.lanes, e.batch)] = e;
        }
        if (!entries_.empty()) save_locked();
    }

    void load()
    {
        load_v2();
        import_legacy_v1();
    }

    void save_locked()
    {
        try {
            const auto path = cache_path();
            std::filesystem::create_directories(path.parent_path());
            std::ofstream out(path, std::ios::trunc);
            if (!out) return;
            out << "YERBAS_CPU_FP 2\n";
            out << std::setprecision(12);
            for (const auto& pair : entries_) {
                const auto& e = pair.second;
                out << std::hex << e.fingerprint << std::dec << ' '
                    << std::quoted(e.policy) << ' '
                    << e.threads << ' ' << e.lanes << ' ' << e.batch << ' '
                    << e.samples << ' ' << e.ewma_hps << ' ' << e.best_hps << ' '
                    << std::quoted(e.cn) << '\n';
            }
        } catch (...) {}
    }

    mutable std::mutex mutex_;
    std::map<std::string, Entry> entries_;
};

inline Store& store()
{
    // WorkerPool can be a function-static object in Stratum. Keep this cache
    // alive through process teardown so its destructor can safely flush without
    // depending on cross-translation-unit static destruction order.
    static Store* instance = new Store();
    return *instance;
}

} // namespace fingerprint_detail

inline FingerprintRuntimeSummary record_fingerprint_runtime(std::uint64_t fingerprint,
                                                            const std::string& cn,
                                                            const std::string& policy,
                                                            unsigned int threads,
                                                            unsigned int lanes,
                                                            unsigned int batch,
                                                            double hps)
{
    return fingerprint_detail::store().record(fingerprint, cn, policy,
                                               threads, lanes, batch, hps);
}

inline std::vector<FingerprintPolicySummary> fingerprint_policy_summaries(std::uint64_t fingerprint)
{
    return fingerprint_detail::store().policies(fingerprint);
}

inline std::vector<FingerprintPolicySummary> exact_fingerprint_policy_summaries(std::uint64_t fingerprint)
{
    return fingerprint_detail::store().exact_policies(fingerprint);
}

inline std::optional<FingerprintPolicySummary> recommended_fingerprint_policy(
    std::uint64_t fingerprint,
    std::uint64_t minimum_samples = 8,
    double minimum_gain = 1.03)
{
    return fingerprint_detail::store().recommended(fingerprint, minimum_samples, minimum_gain);
}

inline void flush_fingerprint_runtime()
{
    fingerprint_detail::store().flush();
}

} // namespace yerbas::cpu
