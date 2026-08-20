#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace yerbas::cuda::cryptonight {

using Hash256 = std::array<std::uint8_t, 32>;

Hash256 validation_hash(int device_id,
                        std::uint8_t variant,
                        const std::uint8_t* input,
                        std::size_t length,
                        float* kernel_ms = nullptr);

// Return the first 32 bytes of the CryptoNight hash_process()/Keccak-1600
// state before scratchpad expansion. This is intentionally exposed only for
// validator diagnostics so we can locate the first CPU/CUDA divergence.
Hash256 validation_keccak_prefix(int device_id,
                                 const std::uint8_t* input,
                                 std::size_t length);

} // namespace yerbas::cuda::cryptonight
