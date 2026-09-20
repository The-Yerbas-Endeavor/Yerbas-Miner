#include "cuda/cuda_backend.h"
#include "ghostrider/ghostrider.h"
#include "ghostrider_vectors.h"

#include <array>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr const char* kCoreNames[] = {
    "BLAKE-512", "BMW-512", "Groestl-512", "JH-512", "Keccak-512",
    "Skein-512", "Luffa-512", "CubeHash-512", "Shavite-512", "SIMD-512",
    "Echo-512", "Hamsi-512", "Fugue-512", "Shabal-512", "Whirlpool"
};

constexpr const char* kCnNames[] = {
    "CN-Dark", "CN-DarkLite", "CN-Fast", "CN-Lite", "CN-Turtle", "CN-TurtleLite"
};

std::string stage_name(std::uint8_t encoded)
{
    const bool cn = (encoded & yerbas::ghostrider::kCryptoNightStageFlag) != 0;
    const std::uint8_t index = static_cast<std::uint8_t>(encoded & 0x7fU);
    if (cn) return index < 6 ? kCnNames[index] : "CN-?";
    return index < 15 ? kCoreNames[index] : "Core-?";
}

bool env_enabled(const char* name)
{
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' && std::string(value) != "0";
}

int parse_cn_variant(std::string v)
{
    for (char& c : v)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    if (v == "0" || v == "dark" || v == "cn-dark") return 0;
    if (v == "1" || v == "darklite" || v == "dark-lite" || v == "cn-darklite" || v == "cn-dark-lite") return 1;
    if (v == "2" || v == "fast" || v == "cn-fast") return 2;
    if (v == "3" || v == "lite" || v == "cn-lite") return 3;
    if (v == "4" || v == "turtle" || v == "cn-turtle") return 4;
    if (v == "5" || v == "turtlelite" || v == "turtle-lite" || v == "cn-turtlelite" || v == "cn-turtle-lite") return 5;
    return -2;
}

int forced_cn_variant()
{
    const char* value = std::getenv("YERBAS_BENCH_FORCE_CN_VARIANT");
    if (value == nullptr || *value == '\0')
        return env_enabled("YERBAS_BENCH_FORCE_CN_FAST") ? 2 : -1;
    return parse_cn_variant(value);
}

std::array<int, 3> forced_cn_triple()
{
    std::array<int, 3> result{{-1, -1, -1}};
    const char* value = std::getenv("YERBAS_BENCH_FORCE_CN_TRIPLE");
    if (value == nullptr || *value == '\0') return result;

    std::string v(value);
    for (char& ch : v) {
        if (ch == ',' || ch == '/' || ch == '+') ch = ' ';
    }

    std::istringstream in(v);
    std::string token;
    for (std::size_t i = 0; i < result.size(); ++i) {
        if (!(in >> token)) {
            result[0] = -2;
            return result;
        }
        result[i] = parse_cn_variant(token);
        if (result[i] < 0) {
            result[0] = -2;
            return result;
        }
    }
    if (in >> token) result[0] = -2;
    return result;
}

std::vector<std::size_t> default_sizes()
{
    return {256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096};
}

std::vector<std::size_t> benchmark_sizes(int argc, char** argv)
{
    if (argc <= 2) return default_sizes();
    std::vector<std::size_t> sizes;
    for (int i = 2; i < argc; ++i) {
        const unsigned long value = std::strtoul(argv[i], nullptr, 10);
        if (value != 0) sizes.push_back(static_cast<std::size_t>(value));
    }
    return sizes.empty() ? default_sizes() : sizes;
}

} // namespace

int main(int argc, char** argv)
{
    const int devices = yerbas::cuda::device_count();
    if (devices == 0) {
        std::cerr << "No CUDA devices detected\n";
        return 1;
    }

    const int device_id = argc > 1 ? std::atoi(argv[1]) : 0;
    if (device_id < 0 || device_id >= devices) {
        std::cerr << "Invalid CUDA device " << device_id << " (detected " << devices << ")\n";
        return 2;
    }

    const int forced_variant = forced_cn_variant();
    const auto forced_triple = forced_cn_triple();
    if (forced_variant == -2) {
        std::cerr << "Invalid YERBAS_BENCH_FORCE_CN_VARIANT. Use 0..5 or "
                     "dark, darklite, fast, lite, turtle, turtlelite.\n";
        return 4;
    }
    if (forced_triple[0] == -2) {
        std::cerr << "Invalid YERBAS_BENCH_FORCE_CN_TRIPLE. Use three variants, "
                     "for example dark,fast,lite.\n";
        return 5;
    }
    const bool force_triple = forced_triple[0] >= 0;
    if (forced_variant >= 0 && force_triple) {
        std::cerr << "Use either YERBAS_BENCH_FORCE_CN_VARIANT or "
                     "YERBAS_BENCH_FORCE_CN_TRIPLE, not both.\n";
        return 6;
    }
    const bool force_cn = forced_variant >= 0 || force_triple;
    const auto sizes = benchmark_sizes(argc, argv);
    auto header = yerbas::test_vectors::MAINNET_GENESIS_HEADER;
    header[76] = header[77] = header[78] = header[79] = 0;
    const yerbas::ghostrider::Work work{header.data(), header.size()};

    yerbas::cuda::JobDescriptor job{};
    job.header = header;
    job.target_le.fill(0xff);
    job.stages = yerbas::ghostrider::stage_schedule(work);

    if (forced_variant >= 0) {
        const std::uint8_t forced_stage = static_cast<std::uint8_t>(
            yerbas::ghostrider::kCryptoNightStageFlag |
            static_cast<std::uint8_t>(forced_variant));
        for (auto& stage : job.stages) {
            if ((stage & yerbas::ghostrider::kCryptoNightStageFlag) != 0)
                stage = forced_stage;
        }
    } else if (force_triple) {
        std::size_t cn_index = 0;
        for (auto& stage : job.stages) {
            if ((stage & yerbas::ghostrider::kCryptoNightStageFlag) == 0)
                continue;
            stage = static_cast<std::uint8_t>(
                yerbas::ghostrider::kCryptoNightStageFlag |
                static_cast<std::uint8_t>(forced_triple[cn_index++]));
        }
    }

    std::cout << "Yerbas CUDA real-pipeline benchmark\n";
    yerbas::cuda::print_devices();
    std::cout << "Benchmark GPU: " << device_id << "\n";
    if (forced_variant >= 0) {
        std::cout << "Schedule mode: forced " << kCnNames[forced_variant]
                  << " production-selector exercise\n";
        if (env_enabled("YERBAS_CN_GEOMETRY_RETUNE"))
            std::cout << "Geometry mode: full real-batch CN block-size sweep\n";
    } else if (force_triple) {
        std::cout << "Schedule mode: forced mixed CN triple "
                  << kCnNames[forced_triple[0]] << '/'
                  << kCnNames[forced_triple[1]] << '/'
                  << kCnNames[forced_triple[2]] << "\n";
    }
    std::cout << "Schedule:";
    for (std::size_t i = 0; i < job.stages.size(); ++i)
        std::cout << " " << i << ":" << stage_name(job.stages[i]);
    std::cout << "\n\n";

    double best_hps = 0.0;
    std::size_t best_batch = 0;

    for (const std::size_t requested : sizes) {
        try {
            yerbas::cuda::BatchEngine engine(device_id, requested, 1);
            engine.upload_job(job);
            const std::size_t actual = engine.batch_size();

            // A forced CN schedule contains three copies of the selected variant
            // per scan. Four untimed scans are enough for the production kernel
            // selector. A geometry-only retune needs enough real-batch launches
            // to finish the full 3-pass block-size candidate sweep as well.
            const bool geometry_retune = env_enabled("YERBAS_CN_GEOMETRY_RETUNE");
            const int warmup_scans = force_cn ? (geometry_retune ? 15 : 4) : 1;
            for (int warmup = 0; warmup < warmup_scans; ++warmup) {
                const std::uint32_t nonce = static_cast<std::uint32_t>(
                    static_cast<std::uint64_t>(warmup) * actual);
                engine.scan(nonce);
            }

            yerbas::cuda::BatchProfile profile{};
            const auto wall_start = std::chrono::steady_clock::now();
            engine.scan_profiled(static_cast<std::uint32_t>(
                                     static_cast<std::uint64_t>(warmup_scans) * actual),
                                 profile);
            const auto wall_stop = std::chrono::steady_clock::now();
            const double wall_ms = std::chrono::duration<double, std::milli>(wall_stop - wall_start).count();
            const double hps = profile.total_gpu_ms > 0.0F
                ? static_cast<double>(profile.hashes) * 1000.0 / profile.total_gpu_ms
                : 0.0;

            std::cout << "=== requested " << requested << " | actual " << actual << " ===\n"
                      << std::fixed << std::setprecision(3)
                      << "GPU total: " << profile.total_gpu_ms << " ms"
                      << " | wall: " << wall_ms << " ms"
                      << " | throughput: " << std::setprecision(2) << hps << " H/s\n"
                      << std::setprecision(3)
                      << "nonce init: " << profile.nonce_init_ms << " ms"
                      << " | stages: " << profile.stage_total_ms << " ms"
                      << " | candidate scan: " << profile.candidate_ms << " ms\n";

            double cn_ms = 0.0;
            double core_ms = 0.0;
            for (const auto& timing : profile.stages) {
                const bool cn = (timing.encoded_stage & yerbas::ghostrider::kCryptoNightStageFlag) != 0;
                if (cn) cn_ms += timing.milliseconds;
                else core_ms += timing.milliseconds;
                const double pct = profile.stage_total_ms > 0.0F
                    ? 100.0 * timing.milliseconds / profile.stage_total_ms : 0.0;
                std::cout << "  [" << std::setw(2) << timing.stage_index << "] "
                          << std::left << std::setw(14) << stage_name(timing.encoded_stage) << std::right
                          << " " << std::setw(10) << timing.milliseconds << " ms"
                          << "  " << std::setw(6) << std::setprecision(2) << pct << "%\n"
                          << std::setprecision(3);
            }
            std::cout << "Core stages: " << core_ms << " ms"
                      << " | CryptoNight stages: " << cn_ms << " ms\n\n";

            if (hps > best_hps) {
                best_hps = hps;
                best_batch = actual;
            }
        } catch (const std::exception& e) {
            std::cout << "=== requested " << requested << " === FAILED: " << e.what() << "\n\n";
        }
    }

    std::cout << "Best batch: " << best_batch << " | " << std::fixed << std::setprecision(2)
              << best_hps << " H/s\n";
    return best_batch == 0 ? 3 : 0;
}
