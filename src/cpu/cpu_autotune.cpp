#include "cpu/cpu_autotune.h"

#include "cpu/cpu_worker_pool.h"

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
#include <string>
#include <vector>

namespace yerbas::cpu {
namespace {

constexpr int kCpuTuneRevision = 2;
constexpr double kCpuTuneMinimumWin = 1.01;

bool env_enabled(const char* name)
{
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' && std::string(value) != "0";
}

bool diagnostics_enabled()
{
    return env_enabled("YERBAS_DIAGNOSTICS");
}

std::uint64_t fnv1a64(const std::string& text)
{
    std::uint64_t value = 14695981039346656037ULL;
    for (const unsigned char c : text) {
        value ^= static_cast<std::uint64_t>(c);
        value *= 1099511628211ULL;
    }
    return value;
}

std::string cpu_identity(unsigned int hardware_threads)
{
    std::ostringstream id;
    id << "threads=" << hardware_threads;
#ifdef YERBAS_NATIVE_CPU_BUILD
    id << "|build=native";
#else
    id << "|build=portable";
#endif
#ifdef _WIN32
    if (const char* p = std::getenv("PROCESSOR_IDENTIFIER")) id << "|cpu=" << p;
#else
    std::ifstream in("/proc/cpuinfo");
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("model name", 0) == 0 || line.rfind("Hardware", 0) == 0) {
            const auto colon = line.find(':');
            if (colon != std::string::npos) id << "|cpu=" << line.substr(colon + 1);
            break;
        }
    }
#endif
#if defined(__GNUC__)
    id << "|gcc=" << __GNUC__ << '.' << __GNUC_MINOR__;
#elif defined(_MSC_VER)
    id << "|msvc=" << _MSC_VER;
#elif defined(__clang__)
    id << "|clang=" << __clang_major__ << '.' << __clang_minor__;
#endif
    return id.str();
}

std::filesystem::path cache_directory()
{
#ifdef _WIN32
    if (const char* local = std::getenv("LOCALAPPDATA"))
        return std::filesystem::path(local) / "Yerbas-Miner" / "cache";
    if (const char* home = std::getenv("USERPROFILE"))
        return std::filesystem::path(home) / ".cache" / "yerbas-miner";
#else
    if (const char* xdg = std::getenv("XDG_CACHE_HOME"))
        return std::filesystem::path(xdg) / "yerbas-miner";
    if (const char* home = std::getenv("HOME"))
        return std::filesystem::path(home) / ".cache" / "yerbas-miner";
#endif
    return std::filesystem::path(".") / ".yerbas-miner-cache";
}

std::filesystem::path cache_path(unsigned int hardware_threads, unsigned int max_threads)
{
    std::ostringstream name;
    name << "cpu-prod-rev" << kCpuTuneRevision << '-'
         << std::hex << fnv1a64(cpu_identity(hardware_threads))
         << "-max" << std::dec << max_threads << ".txt";
    return cache_directory() / name.str();
}

std::array<std::array<std::uint8_t, 80>, 5> representative_headers()
{
    std::array<std::array<std::uint8_t, 80>, 5> headers{};
    for (std::size_t h = 0; h < headers.size(); ++h) {
        auto& header = headers[h];
        header[0] = 4;
        for (std::size_t i = 4; i < 36; ++i)
            header[i] = static_cast<std::uint8_t>((i * (17U + h * 6U) + 29U + h * 41U) & 0xffU);
        for (std::size_t i = 36; i < 76; ++i)
            header[i] = static_cast<std::uint8_t>((i * (11U + h * 4U) + 7U + h * 23U) & 0xffU);
    }
    return headers;
}

void write_nonce(std::array<std::uint8_t, 80>& header, std::uint32_t nonce)
{
    header[76] = static_cast<std::uint8_t>(nonce);
    header[77] = static_cast<std::uint8_t>(nonce >> 8);
    header[78] = static_cast<std::uint8_t>(nonce >> 16);
    header[79] = static_cast<std::uint8_t>(nonce >> 24);
}

double median(std::vector<double> values)
{
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t mid = values.size() / 2U;
    if ((values.size() & 1U) != 0U) return values[mid];
    return (values[mid - 1U] + values[mid]) * 0.5;
}

double benchmark_pair(unsigned int threads, unsigned int batch)
{
    WorkerPool pool(threads);
    const auto headers = representative_headers();
    std::array<std::uint8_t, 32> impossible_target{};

    // Warm the worker pool, CryptoNight reusable contexts and instruction/data
    // caches before collecting production timing samples.
    auto warm = headers.front();
    write_nonce(warm, 0x13572468U);
    (void)pool.run(warm, impossible_target, 0x13572468U, std::max(1U, std::min(batch, 4U)));

    std::vector<double> samples;
    samples.reserve(headers.size());
    std::uint32_t nonce = 0x24680000U;
    for (auto header : headers) {
        write_nonce(header, nonce);
        const auto start = std::chrono::steady_clock::now();
        (void)pool.run(header, impossible_target, nonce, batch);
        const auto stop = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(stop - start).count();
        const double hashes = static_cast<double>(threads) * static_cast<double>(batch);
        if (seconds > 0.0) samples.push_back(hashes / seconds);
        nonce += static_cast<std::uint32_t>(threads * batch + 0x100U);
    }
    return median(std::move(samples));
}

std::vector<unsigned int> batch_candidates(unsigned int configured_batch)
{
    std::set<unsigned int> values{8U, 12U, 16U, 20U, 24U, 28U, 32U, 40U, 48U, 64U};
    if (configured_batch > 0U && configured_batch <= 64U) values.insert(configured_batch);
    return {values.begin(), values.end()};
}

bool load_cache(const std::filesystem::path& path, TuneResult& result)
{
    if (env_enabled("YERBAS_CPU_RETUNE")) return false;
    std::ifstream in(path);
    std::string magic;
    int revision = 0;
    if (!(in >> magic >> revision >> result.threads >> result.batch >> result.throughput_hps)) return false;
    if (magic != "YERBAS_CPU_PROD" || revision != kCpuTuneRevision || result.threads == 0 || result.batch == 0)
        return false;
    result.from_cache = true;
    return true;
}

void save_cache(const std::filesystem::path& path, const TuneResult& result)
{
    try {
        std::filesystem::create_directories(path.parent_path());
        const auto temp = path.string() + ".tmp";
        std::ofstream out(temp, std::ios::trunc);
        if (!out) return;
        out << "YERBAS_CPU_PROD " << kCpuTuneRevision << ' '
            << result.threads << ' ' << result.batch << ' '
            << std::setprecision(12) << result.throughput_hps << '\n';
        out.close();
        std::error_code ec;
        std::filesystem::rename(temp, path, ec);
        if (ec) {
            std::filesystem::remove(path, ec);
            ec.clear();
            std::filesystem::rename(temp, path, ec);
        }
    } catch (...) {
        // Cache persistence is optional; mining remains valid without it.
    }
}

} // namespace

TuneResult production_autotune(unsigned int hardware_threads,
                               unsigned int configured_threads,
                               unsigned int configured_batch)
{
    hardware_threads = std::max(1U, hardware_threads);
    const unsigned int max_threads = configured_threads == 0U
        ? hardware_threads
        : std::max(1U, std::min(configured_threads, hardware_threads));
    const unsigned int fallback_batch = configured_batch == 0U ? 16U : configured_batch;

    if (env_enabled("YERBAS_CPU_DISABLE_AUTOTUNE"))
        return TuneResult{max_threads, fallback_batch, 0.0, false};

    TuneResult cached{};
    const auto path = cache_path(hardware_threads, max_threads);
    if (load_cache(path, cached) && cached.threads <= max_threads) {
        std::cout << "[CPU autotune] cached | hardware_threads=" << hardware_threads
                  << " | threads=" << cached.threads << '/' << max_threads
                  << " | batch=" << cached.batch
                  << " | throughput=" << std::fixed << std::setprecision(2)
                  << cached.throughput_hps << " H/s" << std::defaultfloat << '\n';
        return cached;
    }

    const auto batches = batch_candidates(fallback_batch);
    const std::size_t matrix_size = static_cast<std::size_t>(max_threads) * batches.size();
    std::cout << "[CPU autotune] production matrix tuning starting"
              << " | hardware_threads=" << hardware_threads
              << " | thread_ceiling=" << max_threads
              << " | batches=" << batches.size()
              << " | combinations=" << matrix_size
              << " | schedules=5 | metric=median GhostRider H/s\n";

    TuneResult best{1U, batches.front(), 0.0, false};
    bool have_best = false;

    for (unsigned int threads = 1U; threads <= max_threads; ++threads) {
        TuneResult row_best{threads, batches.front(), 0.0, false};
        for (const unsigned int batch : batches) {
            const double hps = benchmark_pair(threads, batch);
            if (diagnostics_enabled()) {
                std::cout << "[CPU autotune test] threads=" << threads
                          << " | batch=" << batch << " | "
                          << std::fixed << std::setprecision(2) << hps << " H/s"
                          << std::defaultfloat << '\n';
            }
            if (hps > row_best.throughput_hps) {
                row_best.batch = batch;
                row_best.throughput_hps = hps;
            }
        }

        std::cout << "[CPU autotune] threads=" << threads
                  << " | best_batch=" << row_best.batch
                  << " | median=" << std::fixed << std::setprecision(2)
                  << row_best.throughput_hps << " H/s"
                  << std::defaultfloat << '\n';

        if (!have_best || row_best.throughput_hps > best.throughput_hps * kCpuTuneMinimumWin) {
            best = row_best;
            have_best = true;
        }
    }

    save_cache(path, best);
    std::cout << "[CPU autotune] selected"
              << " | hardware_threads=" << hardware_threads
              << " | threads=" << best.threads << '/' << max_threads
              << " | batch=" << best.batch
              << " | median=" << std::fixed << std::setprecision(2)
              << best.throughput_hps << " H/s"
              << " | min-win=1% | cached=yes"
              << std::defaultfloat << '\n';
    return best;
}

} // namespace yerbas::cpu
