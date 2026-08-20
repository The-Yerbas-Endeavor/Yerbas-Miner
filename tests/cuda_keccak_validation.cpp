#include "cuda/cuda_backend.h"
#include "cuda/cryptonight/cn_validation.h"
#include "ghostrider/ghostrider.h"
#include "ghostrider_vectors.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>

namespace {
using Hook = yerbas::cuda::Hash512 (*)(int, const std::uint8_t*, std::size_t);

struct ValidationCase {
    std::uint8_t algorithm;
    const char* name;
    Hook hook;
};

constexpr ValidationCase kCases[] = {
    {0, "BLAKE-512",    yerbas::cuda::blake512_reference_stage},
    {1, "BMW-512",      yerbas::cuda::bmw512_reference_stage},
    {2, "Groestl-512",  yerbas::cuda::groestl512_reference_stage},
    {3, "JH-512",       yerbas::cuda::jh512_reference_stage},
    {4, "Keccak-512",   yerbas::cuda::keccak512_reference_stage},
    {5, "Skein-512",    yerbas::cuda::skein512_reference_stage},
    {6, "Luffa-512",    yerbas::cuda::luffa512_reference_stage},
    {7, "CubeHash-512", yerbas::cuda::cubehash512_reference_stage},
    {8, "Shavite-512",  yerbas::cuda::shavite512_reference_stage},
    {9, "SIMD-512",     yerbas::cuda::simd512_reference_stage},
    {10, "Echo-512",    yerbas::cuda::echo512_reference_stage},
    {11, "Hamsi-512",   yerbas::cuda::hamsi512_reference_stage},
    {12, "Fugue-512",   yerbas::cuda::fugue512_reference_stage},
    {13, "Shabal-512",  yerbas::cuda::shabal512_reference_stage},
    {14, "Whirlpool",   yerbas::cuda::whirlpool512_reference_stage},
};

constexpr const char* kCnNames[] = {
    "CN-Dark", "CN-DarkLite", "CN-Fast", "CN-Lite", "CN-Turtle", "CN-TurtleLite"
};
}

int main()
{
    if (yerbas::cuda::device_count() == 0) {
        std::cout << "CUDA core validation skipped: no CUDA device available\n";
        return 0;
    }

    const auto& header = yerbas::test_vectors::MAINNET_GENESIS_HEADER;
    const yerbas::ghostrider::Work header_work{header.data(), header.size()};
    for (const auto& item : kCases) {
        const auto cpu = yerbas::ghostrider::core_hash_reference(header_work, item.algorithm);
        const auto gpu = item.hook(0, header.data(), header.size());
        if (cpu != gpu) {
            std::cerr << "CUDA " << item.name << " mismatch for 80-byte Yerbas header\n";
            return 10 + item.algorithm;
        }
    }

    std::array<std::uint8_t, 64> stage_input{};
    for (std::size_t i = 0; i < stage_input.size(); ++i)
        stage_input[i] = static_cast<std::uint8_t>((i * 37U + 11U) & 0xffU);
    const yerbas::ghostrider::Work stage_work{stage_input.data(), stage_input.size()};

    for (const auto& item : kCases) {
        const auto cpu = yerbas::ghostrider::core_hash_reference(stage_work, item.algorithm);
        const auto gpu = item.hook(0, stage_input.data(), stage_input.size());
        if (cpu != gpu) {
            std::cerr << "CUDA " << item.name << " mismatch for 64-byte intermediate state\n";
            return 30 + item.algorithm;
        }
    }
    std::cout << "CUDA cores 0-14 match pinned Yerbas Core for 80-byte and 64-byte inputs\n";

    std::cout << std::fixed << std::setprecision(3);
    for (std::uint8_t variant = 0; variant < 6; ++variant) {
        std::cout << "Validating " << kCnNames[variant] << "..." << std::flush;
        const std::uint8_t encoded_stage = static_cast<std::uint8_t>(
            yerbas::ghostrider::kCryptoNightStageFlag | variant);

        const auto cpu_start = std::chrono::steady_clock::now();
        const auto cpu = yerbas::ghostrider::stage_reference(stage_work, encoded_stage);
        const auto cpu_stop = std::chrono::steady_clock::now();
        const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_stop - cpu_start).count();

        float kernel_ms = 0.0F;
        const auto gpu_start = std::chrono::steady_clock::now();
        const auto gpu = yerbas::cuda::cryptonight::validation_hash(
            0, variant, stage_input.data(), stage_input.size(), &kernel_ms);
        const auto gpu_stop = std::chrono::steady_clock::now();
        const double gpu_total_ms = std::chrono::duration<double, std::milli>(gpu_stop - gpu_start).count();

        bool match = true;
        for (std::size_t i = 0; i < gpu.size(); ++i) {
            if (gpu[i] != cpu[i]) { match = false; break; }
        }
        if (!match) {
            std::cerr << " FAILED\nCUDA " << kCnNames[variant]
                      << " mismatch for 64-byte intermediate state\n";
            return 60 + variant;
        }
        std::cout << " OK | kernel " << kernel_ms << " ms"
                  << " | GPU total " << gpu_total_ms << " ms"
                  << " | CPU " << cpu_ms << " ms\n";
    }

    std::cout << "CUDA CryptoNight variants 0-5 match pinned Yerbas Core\n";
    std::cout << "CryptoNight profiling baseline complete; kernel time excludes allocation/copy overhead.\n";
    return 0;
}
