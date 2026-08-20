#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace yerbas::cuda::cryptonight {

using Hash256 = std::array<std::uint8_t, 32>;

struct ValidationCheckpoints {
    std::array<std::uint8_t, 64> expanded_key_prefix{};
    std::array<std::uint8_t, 128> scratchpad_prefix{};
    std::array<std::uint8_t, 80> first_loop_state{};
    std::array<std::uint8_t, 80> second_loop_state{};
    std::array<std::uint8_t, 80> loop16_state{};
    std::array<std::uint8_t, 80> loop1024_state{};
    std::array<std::uint8_t, 80> final_loop_state{};
    std::array<std::uint8_t, 128> collapsed_text{};
    std::array<std::uint8_t, 200> post_keccak_state{};
    std::array<std::uint8_t, 32> final_extra_hash{};
    std::uint8_t extra_hash_selector{0};
};

Hash256 validation_hash(int device_id,
                        std::uint8_t variant,
                        const std::uint8_t* input,
                        std::size_t length,
                        float* kernel_ms = nullptr);

Hash256 validation_keccak_prefix(int device_id,
                                 const std::uint8_t* input,
                                 std::size_t length);

ValidationCheckpoints validation_checkpoints(int device_id,
                                             std::uint8_t variant,
                                             const std::uint8_t* input,
                                             std::size_t length);

} // namespace yerbas::cuda::cryptonight
