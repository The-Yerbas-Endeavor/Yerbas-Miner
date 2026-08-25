#pragma once

#include <array>
#include <algorithm>
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

struct StageProfileSummary {
    std::uint64_t samples{0};
    double total_ewma_ms{0.0};
    std::array<double,18> stage_ewma_ms{};
};

namespace stage_profile_detail {

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
    return cache_dir() / "cpu-stage-profile-rev1.txt";
}

struct Entry {
    std::uint64_t fingerprint{0};
    std::uint64_t samples{0};
    double total_ewma_ms{0.0};
    std::array<double,18> stage_ewma_ms{};
    std::array<std::uint8_t,18> schedule{};
};

class Store {
public:
    Store() { load(); }

    StageProfileSummary record(std::uint64_t fingerprint,
                               const std::array<std::uint8_t,18>& schedule,
                               const std::array<double,18>& stage_ms)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& e = entries_[fingerprint];
        double total = 0.0;
        for (const double v : stage_ms) total += v;
        if (e.samples == 0) {
            e.fingerprint = fingerprint;
            e.schedule = schedule;
            e.total_ewma_ms = total;
            e.stage_ewma_ms = stage_ms;
        } else {
            constexpr double alpha = 0.20;
            e.total_ewma_ms = e.total_ewma_ms * (1.0 - alpha) + total * alpha;
            for (std::size_t i = 0; i < e.stage_ewma_ms.size(); ++i)
                e.stage_ewma_ms[i] = e.stage_ewma_ms[i] * (1.0 - alpha) + stage_ms[i] * alpha;
        }
        ++e.samples;
        if ((e.samples % 8U) == 0U) save_locked();
        return {e.samples, e.total_ewma_ms, e.stage_ewma_ms};
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
        if (!(in >> magic >> rev) || magic != "YERBAS_CPU_STAGE" || rev != 1) return;
        Entry e;
        while (in >> std::hex >> e.fingerprint >> std::dec >> e.samples >> e.total_ewma_ms) {
            for (auto& b : e.schedule) {
                unsigned int v = 0;
                if (!(in >> v)) return;
                b = static_cast<std::uint8_t>(v);
            }
            for (double& v : e.stage_ewma_ms) if (!(in >> v)) return;
            entries_[e.fingerprint] = e;
        }
    }

    void save_locked()
    {
        try {
            const auto path = cache_path();
            std::filesystem::create_directories(path.parent_path());
            std::ofstream out(path, std::ios::trunc);
            if (!out) return;
            out << "YERBAS_CPU_STAGE 1\n" << std::setprecision(12);
            for (const auto& pair : entries_) {
                const auto& e = pair.second;
                out << std::hex << e.fingerprint << std::dec << ' '
                    << e.samples << ' ' << e.total_ewma_ms;
                for (const auto b : e.schedule) out << ' ' << static_cast<unsigned int>(b);
                for (const double v : e.stage_ewma_ms) out << ' ' << v;
                out << '\n';
            }
        } catch (...) {}
    }

    std::mutex mutex_;
    std::map<std::uint64_t, Entry> entries_;
};

inline Store& store()
{
    static Store instance;
    return instance;
}

} // namespace stage_profile_detail

inline StageProfileSummary record_stage_profile(std::uint64_t fingerprint,
                                                const std::array<std::uint8_t,18>& schedule,
                                                const std::array<double,18>& stage_ms)
{
    return stage_profile_detail::store().record(fingerprint, schedule, stage_ms);
}

inline void flush_stage_profiles()
{
    stage_profile_detail::store().flush();
}

} // namespace yerbas::cpu
