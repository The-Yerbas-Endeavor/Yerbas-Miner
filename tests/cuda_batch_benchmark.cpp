#include "cuda/cuda_backend.h"
#include "ghostrider/ghostrider.h"
#include "ghostrider_vectors.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
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

    const auto sizes = benchmark_sizes(argc, argv);
    auto header = yerbas::test_vectors::MAINNET_GENESIS_HEADER;
    header[76] = header[77] = header[78] = header[79] = 0;
    const yerbas::ghostrider::Work work{header.data(), header.size()};

    yerbas::cuda::JobDescriptor job{};
    job.header = header;
    job.target_le.fill(0xff);
    job.stages = yerbas::ghostrider::stage_schedule(work);

    std::cout << "Yerbas CUDA real-pipeline benchmark\n";
    yerbas::cuda::print_devices();
    std::cout << "Benchmark GPU: " << device_id << "\n";
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

            // One untimed warmup removes first-launch/JIT effects from the profile.
            engine.scan(0);

            yerbas::cuda::BatchProfile profile{};
            const auto wall_start = std::chrono::steady_clock::now();
            engine.scan_profiled(static_cast<std::uint32_t>(actual), profile);
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
