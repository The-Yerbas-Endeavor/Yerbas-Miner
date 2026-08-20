#include "cuda/cuda_backend.h"
#include "cuda/cryptonight/cn_validation.h"
#include "ghostrider/ghostrider.h"
#include "ghostrider_vectors.h"
#include "slow-hash.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>

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

template <typename Container>
std::string hex_string(const Container& value)
{
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (const auto byte : value)
        out << std::setw(2) << static_cast<unsigned int>(byte);
    return out.str();
}
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

    // Checkpoint 1: compare the exact 32-byte prefix produced by Core's
    // cn_fast_hash()/hash_process() against CUDA's Keccak-1600 state before
    // any AES expansion or scratchpad work occurs. If this fails, the bug is
    // in Keccak/state initialization; if it passes, the first divergence is
    // later in the CryptoNight slow-hash pipeline.
    std::array<std::uint8_t, 32> cpu_keccak{};
    crypto::cryptonight_dark_fast_hash(
        reinterpret_cast<const char*>(stage_input.data()),
        reinterpret_cast<char*>(cpu_keccak.data()),
        static_cast<std::uint32_t>(stage_input.size()));
    const auto gpu_keccak = yerbas::cuda::cryptonight::validation_keccak_prefix(
        0, stage_input.data(), stage_input.size());

    if (cpu_keccak != gpu_keccak) {
        std::cerr << "CryptoNight checkpoint FAILED: initial Keccak/hash_process state diverges\n"
                  << "  CPU: " << hex_string(cpu_keccak) << "\n"
                  << "  GPU: " << hex_string(gpu_keccak) << "\n";
        return 55;
    }
    std::cout << "CryptoNight checkpoint OK: initial Keccak/hash_process state matches Core\n";

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
                      << " mismatch for 64-byte intermediate state\n"
                      << "  CPU final: " << hex_string(cpu) << "\n"
                      << "  GPU final: " << hex_string(gpu) << "\n"
                      << "  Initial Keccak checkpoint already matched; divergence is after hash_process\n";
            return 60 + variant;
        }
        std::cout << " OK | kernel " << kernel_ms << " ms"
                  << " | GPU total " << gpu_total_ms << " ms"
                  << " | CPU " << cpu_ms << " ms\n";
    }

    std::cout << "CUDA CryptoNight variants 0-5 match pinned Yerbas Core\n";
    std::cout << "CryptoNight profiling baseline complete; kernel time excludes allocation/copy overhead.\n";

    std::array<std::uint8_t, 80> full_header = header;
    full_header[76] = 0;
    full_header[77] = 0;
    full_header[78] = 0;
    full_header[79] = 0;
    const yerbas::ghostrider::Work full_work{full_header.data(), full_header.size()};
    const auto cpu_full = yerbas::ghostrider::hash_reference(full_work);

    yerbas::cuda::JobDescriptor job{};
    job.header = full_header;
    job.target_le.fill(0xff);
    job.stages = yerbas::ghostrider::stage_schedule(full_work);

    yerbas::cuda::BatchEngine engine(0, 1);
    engine.upload_job(job);
    const auto candidates = engine.scan(0);
    if (candidates.empty() || candidates.front().nonce != 0) {
        std::cerr << "CUDA full-pipeline validation failed to return nonce 0\n";
        return 80;
    }
    if (candidates.front().hash != cpu_full) {
        std::cerr << "CUDA full GhostRider pipeline DOES NOT match CPU reference for nonce 0\n"
                  << "  CPU: " << hex_string(cpu_full) << "\n"
                  << "  GPU: " << hex_string(candidates.front().hash) << "\n";
        return 81;
    }
    std::cout << "CUDA full 18-stage GhostRider pipeline matches CPU reference for nonce 0\n";
    return 0;
}
