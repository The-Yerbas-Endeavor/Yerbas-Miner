#pragma once

#include "config.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace yerbas {
namespace first_run {

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

inline bool interactive_stdin()
{
#ifdef _WIN32
    return _isatty(_fileno(stdin)) != 0;
#else
    return isatty(fileno(stdin)) != 0;
#endif
}

inline bool cache_has_prefix(const std::string& prefix)
{
    std::error_code ec;
    const auto dir = cache_dir();
    if (!std::filesystem::exists(dir, ec)) return false;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file(ec)) continue;
        const std::string name = entry.path().filename().string();
        if (name.rfind(prefix, 0) == 0) return true;
    }
    return false;
}

inline void remember_decline()
{
    try {
        std::filesystem::create_directories(cache_dir());
        std::ofstream out(cache_dir() / "autotune-declined", std::ios::trunc);
        if (out) out << "1\n";
    } catch (...) {}
}

inline bool declined_before()
{
    std::error_code ec;
    return std::filesystem::exists(cache_dir() / "autotune-declined", ec);
}

inline void clear_decline_marker()
{
    std::error_code ec;
    std::filesystem::remove(cache_dir() / "autotune-declined", ec);
}

inline void apply(AppConfig& cfg)
{
    // Explicit calibration flags always win and never prompt.
    if (cfg.miner.autotune || cfg.gpu.autotune) return;

    const bool cpu_profile = !cfg.miner.cpu_enabled || cache_has_prefix("cpu-policy-rev");
    const bool gpu_profile = !cfg.gpu.enabled || cache_has_prefix("gpu-calibration-rev");
    if (cpu_profile && gpu_profile) return;

    // A remembered No means safe immediate startup, not a repeated prompt.
    if (declined_before()) {
        if (!cpu_profile) cfg.miner.cpu_tune = "off";
        std::cout << "[First run] hardware autotuning previously declined | using saved/safe settings\n";
        return;
    }

    // Services, pipes, cron, and other non-interactive launches must never block.
    if (!interactive_stdin()) {
        if (!cpu_profile) cfg.miner.cpu_tune = "off";
        std::cout << "[First run] no tuning profile | non-interactive startup uses safe settings\n";
        return;
    }

    std::cout << "\n🌿 Yerbas Miner — First Run\n\n"
              << "No saved hardware tuning profile was found for this machine.\n\n"
              << "Hardware autotuning benchmarks your CPU and GPU and selects\n"
              << "settings optimized for this system. Progress is shown while it runs.\n\n"
              << "Run hardware autotuning now? [Y/n]: " << std::flush;

    std::string answer;
    std::getline(std::cin, answer);
    const bool yes = answer.empty() || answer == "y" || answer == "Y" ||
                     answer == "yes" || answer == "YES" || answer == "Yes";

    if (yes) {
        clear_decline_marker();
        cfg.miner.autotune = cfg.miner.cpu_enabled;
        cfg.gpu.autotune = cfg.gpu.enabled;
        if (cfg.miner.cpu_enabled) cfg.miner.cpu_tune = "default";
        std::cout << "[First run] hardware autotuning selected\n\n";
    } else {
        remember_decline();
        if (!cpu_profile) cfg.miner.cpu_tune = "off";
        cfg.gpu.autotune = false;
        std::cout << "[First run] autotuning skipped | starting with safe automatic settings\n\n";
    }
}

} // namespace first_run
} // namespace yerbas
