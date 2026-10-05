#pragma once

#include "config.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <conio.h>
#include <io.h>
#include <windows.h>
#else
#include <sys/select.h>
#include <unistd.h>
#endif

namespace yerbas {
namespace first_run {

#ifdef _WIN32
inline std::filesystem::path executable_dir()
{
    std::array<char, 32768> buffer{};
    const DWORD len = GetModuleFileNameA(
        nullptr,
        buffer.data(),
        static_cast<DWORD>(buffer.size()));
    if (len == 0 || len >= buffer.size())
        return std::filesystem::current_path();
    return std::filesystem::path(
        std::string(buffer.data(), len)).parent_path();
}
#endif

inline std::filesystem::path cache_dir()
{
#ifdef _WIN32
    if (const char* p = std::getenv("LOCALAPPDATA"); p && *p)
        return std::filesystem::path(p) / "Yerbas-Miner" / "cache";
    if (const char* p = std::getenv("USERPROFILE"); p && *p)
        return std::filesystem::path(p) / ".cache" / "yerbas-miner";
#else
    if (const char* p = std::getenv("XDG_CACHE_HOME"); p && *p)
        return std::filesystem::path(p) / "yerbas-miner";
    if (const char* p = std::getenv("HOME"); p && *p)
        return std::filesystem::path(p) / ".cache" / "yerbas-miner";
#endif
    return std::filesystem::path(".") / ".yerbas-miner-cache";
}

inline std::vector<std::filesystem::path> cache_search_dirs()
{
    std::vector<std::filesystem::path> dirs;
    dirs.push_back(cache_dir());
#ifdef _WIN32
    if (const char* p = std::getenv("LOCALAPPDATA"); p && *p) {
        const auto legacy = std::filesystem::path(p) / "Yerbas-Miner";
        if (legacy != dirs.front()) dirs.push_back(legacy);
    }
    // Portable Windows builds often keep tuning files beside the executable.
    // Use the executable directory, not just the process working directory:
    // shortcuts and shells may start the miner from somewhere else.
    const auto exe_dir = executable_dir();
    bool seen = false;
    for (const auto& dir : dirs) {
        if (dir == exe_dir) {
            seen = true;
            break;
        }
    }
    if (!seen) dirs.push_back(exe_dir);

    const auto cwd = std::filesystem::current_path();
    seen = false;
    for (const auto& dir : dirs) {
        if (dir == cwd) {
            seen = true;
            break;
        }
    }
    if (!seen) dirs.push_back(cwd);
#endif
    return dirs;
}

inline bool interactive_stdin()
{
#ifdef _WIN32
    return _isatty(_fileno(stdin)) != 0;
#else
    return isatty(fileno(stdin)) != 0;
#endif
}

inline bool prompt_autotune_with_timeout(unsigned int timeout_seconds)
{
    std::cout << "Run missing hardware autotuning now? [Y/n]\n"
              << "Autotuning will start automatically in "
              << timeout_seconds << " seconds if nothing is selected.\n"
              << std::flush;

#ifdef _WIN32
    for (unsigned int remaining = timeout_seconds; remaining > 0U; --remaining) {
        std::cout << "\rAuto-starting in " << remaining << "s... "
                  << "(Y/Enter = tune, N = skip)   " << std::flush;

        const DWORD slice_ms = 1000U;
        const DWORD start = GetTickCount();
        while (GetTickCount() - start < slice_ms) {
            if (_kbhit()) {
                const int ch = _getch();
                if (ch == 'n' || ch == 'N') {
                    std::cout << "\rAutotuning skipped by user.                              \n";
                    return false;
                }
                if (ch == 'y' || ch == 'Y' || ch == '\r' || ch == '\n') {
                    std::cout << "\rAutotuning selected.                                    \n";
                    return true;
                }
            }
            Sleep(25);
        }
    }
#else
    for (unsigned int remaining = timeout_seconds; remaining > 0U; --remaining) {
        std::cout << "\rAuto-starting in " << remaining
                  << "s... (press Enter for default, or type n + Enter to skip)   "
                  << std::flush;

        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(STDIN_FILENO, &readfds);
        timeval timeout{};
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;

        const int ready = select(STDIN_FILENO + 1, &readfds, nullptr, nullptr, &timeout);
        if (ready > 0 && FD_ISSET(STDIN_FILENO, &readfds)) {
            std::string answer;
            std::getline(std::cin, answer);
            const bool yes = answer.empty() || answer == "y" || answer == "Y" ||
                             answer == "yes" || answer == "YES" || answer == "Yes";
            std::cout << '\r'
                      << (yes ? "Autotuning selected."
                              : "Autotuning skipped by user.")
                      << "                                      \n";
            return yes;
        }
        if (ready < 0) break;
    }
#endif

    std::cout << "\rNo selection received — starting hardware autotuning now.           \n";
    return true;
}

inline bool cache_has_prefix(const std::string& prefix)
{
    for (const auto& dir : cache_search_dirs()) {
        std::error_code ec;
        if (!std::filesystem::exists(dir, ec)) continue;
        for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
            if (ec) break;
            if (!entry.is_regular_file(ec)) continue;
            const std::string name = entry.path().filename().string();
            if (name.rfind(prefix, 0) == 0) return true;
        }
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
    // Explicit calibration flags and explicit full GPU tuning always win and
    // never prompt. "off" likewise means the operator has already decided not
    // to request a first-run GPU benchmark.
    if (cfg.miner.autotune || cfg.gpu.autotune || cfg.gpu.gpu_tune == "full") return;

    const bool cpu_profile = !cfg.miner.cpu_enabled || cache_has_prefix("cpu-policy-rev");
    const bool gpu_profile = !cfg.gpu.enabled || cfg.gpu.gpu_tune == "off" ||
                             cache_has_prefix("gpu-calibration-rev");
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
              << "Missing tuning profile:"
              << (!cpu_profile ? " CPU" : "")
              << (!gpu_profile ? " GPU" : "")
              << "\n\n"
              << "Hardware autotuning benchmarks only the missing component(s) and\n"
              << "keeps any valid saved tuning that is already present.\n\n";

    const bool yes = prompt_autotune_with_timeout(30U);

    if (yes) {
        clear_decline_marker();

        // Tune only what is actually missing. A valid GPU cache must never be
        // discarded just because the CPU profile is absent (or vice versa).
        cfg.miner.autotune =
            cfg.miner.cpu_enabled && !cpu_profile;
        cfg.gpu.autotune =
            cfg.gpu.enabled &&
            cfg.gpu.gpu_tune == "auto" &&
            !gpu_profile;

        if (cfg.miner.autotune)
            cfg.miner.cpu_tune = "default";

        std::cout << "[First run] hardware autotuning selected"
                  << " | CPU=" << (cfg.miner.autotune ? "tune" : "cached")
                  << " | GPU=" << (cfg.gpu.autotune ? "tune" : "cached")
                  << "\n\n";
    } else {
        remember_decline();
        if (!cpu_profile) cfg.miner.cpu_tune = "off";
        cfg.gpu.autotune = false;
        std::cout << "[First run] autotuning skipped | starting with safe automatic settings\n\n";
    }
}

} // namespace first_run
} // namespace yerbas
