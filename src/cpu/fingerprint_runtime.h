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
#include <sstream>
#include <string>

namespace yerbas::cpu {

struct FingerprintRuntimeSummary {
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
    return cache_dir() / "cpu-fingerprint-runtime-rev1.txt";
}

inline std::string key(std::uint64_t fingerprint,
                       unsigned int threads,
                       unsigned int lanes,
                       unsigned int batch)
{
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << fingerprint
        << std::dec << ':' << threads << ':' << lanes << ':' << batch;
    return out.str();
}

struct Entry {
    std::uint64_t fingerprint{0};
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
                                     unsigned int threads,
                                     unsigned int lanes,
                                     unsigned int batch,
                                     double hps)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto k = key(fingerprint, threads, lanes, batch);
        auto& e = entries_[k];
        if (e.samples == 0) {
            e.fingerprint = fingerprint;
            e.threads = threads;
            e.lanes = lanes;
            e.batch = batch;
            e.cn = cn;
            e.ewma_hps = hps;
            e.best_hps = hps;
        } else {
            // Slow enough to reject one-off spikes while still adapting across runs.
            constexpr double alpha = 0.20;
            e.ewma_hps = e.ewma_hps * (1.0 - alpha) + hps * alpha;
            e.best_hps = std::max(e.best_hps, hps);
        }
        ++e.samples;
        if ((e.samples % 8U) == 0U) save_locked();
        return {e.samples, e.ewma_hps, e.best_hps};
    }

    void flush()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        save_locked();
    }

private:
    void load()
    {
        std::ifstream in(cache_path());
        std::string magic;
        int rev = 0;
        if (!(in >> magic >> rev) || magic != "YERBAS_CPU_FP" || rev != 1) return;
        Entry e;
        while (in >> std::hex >> e.fingerprint >> std::dec
                  >> e.threads >> e.lanes >> e.batch >> e.samples
                  >> e.ewma_hps >> e.best_hps >> std::quoted(e.cn)) {
            entries_[key(e.fingerprint, e.threads, e.lanes, e.batch)] = e;
        }
    }

    void save_locked()
    {
        try {
            const auto path = cache_path();
            std::filesystem::create_directories(path.parent_path());
            std::ofstream out(path, std::ios::trunc);
            if (!out) return;
            out << "YERBAS_CPU_FP 1\n";
            out << std::setprecision(12);
            for (const auto& pair : entries_) {
                const auto& e = pair.second;
                out << std::hex << e.fingerprint << std::dec << ' '
                    << e.threads << ' ' << e.lanes << ' ' << e.batch << ' '
                    << e.samples << ' ' << e.ewma_hps << ' ' << e.best_hps << ' '
                    << std::quoted(e.cn) << '\n';
            }
        } catch (...) {}
    }

    std::mutex mutex_;
    std::map<std::string, Entry> entries_;
};

inline Store& store()
{
    static Store instance;
    return instance;
}

} // namespace fingerprint_detail

inline FingerprintRuntimeSummary record_fingerprint_runtime(std::uint64_t fingerprint,
                                                            const std::string& cn,
                                                            unsigned int threads,
                                                            unsigned int lanes,
                                                            unsigned int batch,
                                                            double hps)
{
    return fingerprint_detail::store().record(fingerprint, cn, threads, lanes, batch, hps);
}

inline void flush_fingerprint_runtime()
{
    fingerprint_detail::store().flush();
}

} // namespace yerbas::cpu
