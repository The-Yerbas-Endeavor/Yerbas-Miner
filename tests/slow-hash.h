#pragma once

#include <cstddef>
#include <cstdint>

// Compatibility surface for the pinned Yerbas Core CryptoNight API used by
// cuda_keccak_validation.cpp and the Core c_groestl header.
using BitSequence = unsigned char;
using DataLength = unsigned long long;

constexpr std::uint32_t CN_DARK_PAGE_SIZE = 524288U;
constexpr std::uint32_t CN_DARK_ITERATIONS = 131072U;
constexpr std::size_t CN_DARK_AES_ROUNDS = 32768U;

namespace crypto {
extern "C" {
void cn_fast_hash(const char* input, char* output, std::uint32_t len);
void cn_slow_hash(const char* input,
                  char* output,
                  std::uint32_t len,
                  int variant,
                  std::uint32_t page_size,
                  std::uint32_t iterations,
                  std::size_t aes_rounds);
}

inline void cryptonight_dark_fast_hash(const char* input, char* output, std::uint32_t len)
{
    cn_fast_hash(input, output, len);
}

inline void cryptonight_dark_hash(const char* input,
                                  char* output,
                                  std::uint32_t len,
                                  int variant)
{
    cn_slow_hash(input,
                 output,
                 len,
                 variant,
                 CN_DARK_PAGE_SIZE,
                 CN_DARK_ITERATIONS,
                 CN_DARK_AES_ROUNDS);
}
} // namespace crypto
