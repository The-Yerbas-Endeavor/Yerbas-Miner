#include "cpu/cn_width_tune.h"
#include "cpu/cn_2way.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

extern "C" const char* yerbas_cn_reuse_backend(void);

namespace yerbas::cpu {
namespace {

constexpr int kWidthTuneRevision = 1;
constexpr double kMinimumGain = 1.02;

struct Params {
    const char* name;
    std::uint32_t page;
    std::uint32_t iterations;
    std::size_t aes_rounds;
};

constexpr std::array<Params, 6> kProfiles{{
    {"Dark",       524288U,  131072U,  32768U},
    {"DarkLite",   524288U,  131072U,  16384U},
    {"Fast",      2097152U,  262144U, 131072U},
    {"Lite",      1048576U,  262144U,  65536U},
    {"Turtle",     262144U,   65536U,  16384U},
    {"TurtleLite", 262144U,   65536U,   8192U},
}};

bool env_enabled(const char* name)
{
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' && std::string(value) != "0";
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

std::uint64_t fnv1a(const std::string& s)
{
    std::uint64_t v = 14695981039346656037ULL;
    for (unsigned char c : s) { v ^= c; v *= 1099511628211ULL; }
    return v;
}

std::filesystem::path cache_path()
{
    std::ostringstream key;
    key << yerbas_cn_reuse_backend();
#ifdef YERBAS_NATIVE_CPU_BUILD
    key << "|native";
#else
    key << "|portable";
#endif
    key << "|hw=" << std::thread::hardware_concurrency();
    std::ostringstream name;
    name << "cn-width-rev" << kWidthTuneRevision << '-' << std::hex << fnv1a(key.str()) << ".txt";
    return cache_dir() / name.str();
}

void make_inputs(std::array<std::array<char,64>,4>& inputs)
{
    for (std::size_t lane = 0; lane < inputs.size(); ++lane) {
        for (std::size_t i = 0; i < 64; ++i)
            inputs[lane][i] = static_cast<char>((i * (31U + lane * 6U) + 17U + lane * 43U) & 0xffU);
    }
}

template <class F>
double median_ms(F&& fn, int repeats)
{
    std::vector<double> values;
    values.reserve(static_cast<std::size_t>(repeats));
    for (int i = 0; i < repeats; ++i) {
        const auto start = std::chrono::steady_clock::now();
        fn();
        const auto end = std::chrono::steady_clock::now();
        values.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    std::sort(values.begin(), values.end());
    return values[values.size()/2U];
}

bool load_cache(const std::filesystem::path& path, CnWidthTuneResult& out)
{
    if (env_enabled("YERBAS_CPU_RETUNE") || env_enabled("YERBAS_CPU_WIDTH_RETUNE")) return false;
    std::ifstream in(path);
    std::string magic;
    int rev = 0;
    if (!(in >> magic >> rev)) return false;
    if (magic != "YERBAS_CN_WIDTH" || rev != kWidthTuneRevision) return false;
    for (auto& w : out.widths) {
        if (!(in >> w) || (w != 1U && w != 2U && w != 4U)) return false;
    }
    out.from_cache = true;
    return true;
}

void save_cache(const std::filesystem::path& path, const CnWidthTuneResult& result)
{
    try {
        std::filesystem::create_directories(path.parent_path());
        std::ofstream out(path, std::ios::trunc);
        if (!out) return;
        out << "YERBAS_CN_WIDTH " << kWidthTuneRevision;
        for (const auto w : result.widths) out << ' ' << w;
        out << '\n';
    } catch (...) {}
}

void print_summary(const CnWidthTuneResult& result, const char* source)
{
    std::cout << "CPU CN widths:"
              << " Dark=" << result.widths[0]
              << " DarkLite=" << result.widths[1]
              << " Fast=" << result.widths[2]
              << " Lite=" << result.widths[3]
              << " Turtle=" << result.widths[4]
              << " TurtleLite=" << result.widths[5]
              << " | source=" << source << '\n';
}

} // namespace

CnWidthTuneResult qualify_cn_widths(const std::string& mode)
{
    CnWidthTuneResult result{};
    if (mode == "off" || mode == "simple") {
        print_summary(result, "default");
        return result;
    }

    const auto path = cache_path();
    if (load_cache(path, result)) {
        print_summary(result, "cache");
        return result;
    }

    std::cout << "[CPU CN width tune] qualifying 1-way/2-way/4-way across 6 profiles | parity=required | min-gain=2%\n";

    std::array<std::array<char,64>,4> input{};
    make_inputs(input);

    for (std::size_t p = 0; p < kProfiles.size(); ++p) {
        const auto& profile = kProfiles[p];
        std::array<std::array<char,32>,4> ref{};
        std::array<std::array<char,32>,4> out2{};
        std::array<std::array<char,32>,4> out4{};

        const auto ref2 = [&] {
            yerbas_cn_hash_pair_reference(input[0].data(), input[1].data(), ref[0].data(), ref[1].data(),
                                          64U, 1, profile.page, profile.iterations, profile.aes_rounds);
        };
        const auto ref4 = [&] {
            yerbas_cn_hash_pair_reference(input[0].data(), input[1].data(), ref[0].data(), ref[1].data(),
                                          64U, 1, profile.page, profile.iterations, profile.aes_rounds);
            yerbas_cn_hash_pair_reference(input[2].data(), input[3].data(), ref[2].data(), ref[3].data(),
                                          64U, 1, profile.page, profile.iterations, profile.aes_rounds);
        };
        const auto run2 = [&] {
            yerbas_cn_hash_pair_2way(input[0].data(), input[1].data(), out2[0].data(), out2[1].data(),
                                     64U, 1, profile.page, profile.iterations, profile.aes_rounds);
        };
        const auto run4 = [&] {
            yerbas_cn_hash_quad_4way(input[0].data(), input[1].data(), input[2].data(), input[3].data(),
                                     out4[0].data(), out4[1].data(), out4[2].data(), out4[3].data(),
                                     64U, 1, profile.page, profile.iterations, profile.aes_rounds);
        };

        ref4(); run2(); run4();
        const bool parity2 = std::memcmp(ref[0].data(), out2[0].data(), 32) == 0 &&
                             std::memcmp(ref[1].data(), out2[1].data(), 32) == 0;
        const bool parity4 = parity2 &&
                             std::memcmp(ref[0].data(), out4[0].data(), 32) == 0 &&
                             std::memcmp(ref[1].data(), out4[1].data(), 32) == 0 &&
                             std::memcmp(ref[2].data(), out4[2].data(), 32) == 0 &&
                             std::memcmp(ref[3].data(), out4[3].data(), 32) == 0;

        const double one2_ms = median_ms(ref2, 3);
        const double two_ms = parity2 ? median_ms(run2, 3) : 1e30;
        const double one4_ms = median_ms(ref4, 3);
        const double four_ms = parity4 ? median_ms(run4, 3) : 1e30;

        unsigned int selected = 1;
        double best_per_hash = one2_ms / 2.0;
        const double two_per_hash = two_ms / 2.0;
        const double four_per_hash = four_ms / 4.0;
        if (parity2 && two_per_hash * kMinimumGain < best_per_hash) {
            selected = 2;
            best_per_hash = two_per_hash;
        }
        if (parity4 && four_per_hash * kMinimumGain < best_per_hash) {
            selected = 4;
            best_per_hash = four_per_hash;
        }
        result.widths[p] = selected;

        std::cout << "[CPU CN width] " << profile.name
                  << " | 1way=" << std::fixed << std::setprecision(3) << (one2_ms/2.0) << " ms/hash"
                  << " | 2way=" << (parity2 ? two_per_hash : 0.0) << " ms/hash(" << (parity2 ? "PASS" : "FAIL") << ')'
                  << " | 4way=" << (parity4 ? four_per_hash : 0.0) << " ms/hash(" << (parity4 ? "PASS" : "FAIL") << ')'
                  << " | selected=" << selected << "way" << std::defaultfloat << '\n';
    }

    save_cache(path, result);
    print_summary(result, "fresh");
    return result;
}

} // namespace yerbas::cpu
