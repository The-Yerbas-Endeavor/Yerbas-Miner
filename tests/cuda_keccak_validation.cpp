#include "cuda/cuda_backend.h"
#include "ghostrider/ghostrider.h"
#include "ghostrider_vectors.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>

int main()
{
    if (yerbas::cuda::device_count() == 0) {
        std::cout << "CUDA validation skipped: no CUDA device available\n";
        return 0;
    }

    const auto& header = yerbas::test_vectors::MAINNET_GENESIS_HEADER;
    const yerbas::ghostrider::Work header_work{header.data(), header.size()};

    // Yerbas Core coreHash index 4 == sph_keccak512.
    const auto cpu_header = yerbas::ghostrider::core_hash_reference(header_work, 4);
    const auto gpu_header = yerbas::cuda::keccak512_reference_stage(0, header.data(), header.size());
    if (cpu_header != gpu_header) {
        std::cerr << "CUDA Keccak-512 mismatch for 80-byte Yerbas header\n";
        return 1;
    }

    std::array<std::uint8_t, 64> stage_input{};
    for (std::size_t i = 0; i < stage_input.size(); ++i) {
        stage_input[i] = static_cast<std::uint8_t>((i * 37U + 11U) & 0xffU);
    }
    const yerbas::ghostrider::Work stage_work{stage_input.data(), stage_input.size()};
    const auto cpu_stage = yerbas::ghostrider::core_hash_reference(stage_work, 4);
    const auto gpu_stage = yerbas::cuda::keccak512_reference_stage(0, stage_input.data(), stage_input.size());
    if (cpu_stage != gpu_stage) {
        std::cerr << "CUDA Keccak-512 mismatch for 64-byte intermediate state\n";
        return 2;
    }

    std::cout << "CUDA Keccak-512 matches Yerbas Core for 80-byte and 64-byte inputs\n";
    return 0;
}
