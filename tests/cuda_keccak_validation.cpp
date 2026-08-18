#include "cuda/cuda_backend.h"
#include "ghostrider/ghostrider.h"
#include "ghostrider_vectors.h"

#include <array>
#include <cstdint>
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
    return 0;
}
