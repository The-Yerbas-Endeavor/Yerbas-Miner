#pragma once

#include <cstddef>
#include <cstdint>

namespace yerbas::cuda::core {

__device__ __forceinline__ std::uint64_t skein_rotl64(std::uint64_t x, unsigned n)
{
    return (x << n) | (x >> (64U - n));
}

__device__ __forceinline__ std::uint64_t skein_load_le64(const std::uint8_t* p)
{
    return static_cast<std::uint64_t>(p[0]) |
           (static_cast<std::uint64_t>(p[1]) << 8) |
           (static_cast<std::uint64_t>(p[2]) << 16) |
           (static_cast<std::uint64_t>(p[3]) << 24) |
           (static_cast<std::uint64_t>(p[4]) << 32) |
           (static_cast<std::uint64_t>(p[5]) << 40) |
           (static_cast<std::uint64_t>(p[6]) << 48) |
           (static_cast<std::uint64_t>(p[7]) << 56);
}

__device__ __forceinline__ void skein_store_le64(std::uint8_t* p, std::uint64_t x)
{
    p[0] = static_cast<std::uint8_t>(x);
    p[1] = static_cast<std::uint8_t>(x >> 8);
    p[2] = static_cast<std::uint8_t>(x >> 16);
    p[3] = static_cast<std::uint8_t>(x >> 24);
    p[4] = static_cast<std::uint8_t>(x >> 32);
    p[5] = static_cast<std::uint8_t>(x >> 40);
    p[6] = static_cast<std::uint8_t>(x >> 48);
    p[7] = static_cast<std::uint8_t>(x >> 56);
}

__device__ __forceinline__ void skein_mix(std::uint64_t& a,
                                          std::uint64_t& b,
                                          unsigned rotation)
{
    a += b;
    b = skein_rotl64(b, rotation) ^ a;
}

__device__ __forceinline__ void threefish512_encrypt(const std::uint64_t key[8],
                                                     std::uint64_t tweak0,
                                                     std::uint64_t tweak1,
                                                     const std::uint64_t block[8],
                                                     std::uint64_t out[8])
{
    constexpr std::uint64_t parity = 0x1BD11BDAA9FC1A22ULL;
    const unsigned rotations[8][4] = {
        {46, 36, 19, 37},
        {33, 27, 14, 42},
        {17, 49, 36, 39},
        {44,  9, 54, 56},
        {39, 30, 34, 24},
        {13, 50, 10, 17},
        {25, 29, 39, 43},
        { 8, 35, 56, 22}
    };
    const int permutation[8] = {2, 1, 4, 7, 6, 5, 0, 3};

    std::uint64_t k[9];
    k[8] = parity;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        k[i] = key[i];
        k[8] ^= key[i];
    }
    const std::uint64_t t[3] = {tweak0, tweak1, tweak0 ^ tweak1};

    std::uint64_t x[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) x[i] = block[i];

    // Initial key injection (subkey s=0).
    #pragma unroll
    for (int i = 0; i < 8; ++i) x[i] += k[i];
    x[5] += t[0];
    x[6] += t[1];

    #pragma unroll
    for (int round = 0; round < 72; ++round) {
        const unsigned* r = rotations[round & 7];
        skein_mix(x[0], x[1], r[0]);
        skein_mix(x[2], x[3], r[1]);
        skein_mix(x[4], x[5], r[2]);
        skein_mix(x[6], x[7], r[3]);

        std::uint64_t p[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) p[i] = x[permutation[i]];
        #pragma unroll
        for (int i = 0; i < 8; ++i) x[i] = p[i];

        if ((round & 3) == 3) {
            const int s = (round + 1) >> 2;
            #pragma unroll
            for (int i = 0; i < 8; ++i) x[i] += k[(s + i) % 9];
            x[5] += t[s % 3];
            x[6] += t[(s + 1) % 3];
            x[7] += static_cast<std::uint64_t>(s);
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = x[i];
}

__device__ __forceinline__ void skein512_ubi(std::uint64_t chain[8],
                                             const std::uint8_t block_bytes[64],
                                             std::uint64_t position,
                                             std::uint64_t type,
                                             bool first,
                                             bool final)
{
    std::uint64_t block[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) block[i] = skein_load_le64(block_bytes + i * 8);

    std::uint64_t tweak1 = type << 56;
    if (first) tweak1 |= (1ULL << 62);
    if (final) tweak1 |= (1ULL << 63);

    std::uint64_t encrypted[8];
    threefish512_encrypt(chain, position, tweak1, block, encrypted);
    #pragma unroll
    for (int i = 0; i < 8; ++i) chain[i] = encrypted[i] ^ block[i];
}

// Skein-512-512 as used by sphlib/Yerbas Core. GhostRider feeds either the
// original 80-byte block header or a 64-byte intermediate digest.
__device__ __forceinline__ void skein512(const std::uint8_t* input,
                                         std::size_t length,
                                         std::uint8_t out[64])
{
    // Precomputed Skein-512-512 configuration chaining value.
    std::uint64_t chain[8] = {
        0x4903ADFF749C51CEULL, 0x0D95DE399746DF03ULL,
        0x8FD1934127C79BCEULL, 0x9A255629FF352CB1ULL,
        0x5DB62599DF6CA7B0ULL, 0xEABE394CA9D5C3F4ULL,
        0x991112C71A75B523ULL, 0xAE18A40B660FCC33ULL
    };

    // Message UBI type = 48.
    if (length <= 64) {
        std::uint8_t block[64];
        #pragma unroll
        for (int i = 0; i < 64; ++i) block[i] = 0;
        for (std::size_t i = 0; i < length; ++i) block[i] = input[i];
        skein512_ubi(chain, block, static_cast<std::uint64_t>(length), 48, true, true);
    } else {
        // GhostRider's only longer input is the 80-byte block header.
        std::uint8_t block0[64];
        std::uint8_t block1[64];
        #pragma unroll
        for (int i = 0; i < 64; ++i) {
            block0[i] = input[i];
            block1[i] = 0;
        }
        const std::size_t remaining = length - 64;
        for (std::size_t i = 0; i < remaining && i < 64; ++i) block1[i] = input[64 + i];
        skein512_ubi(chain, block0, 64, 48, true, false);
        skein512_ubi(chain, block1, static_cast<std::uint64_t>(length), 48, false, true);
    }

    // Output UBI type = 63. Skein-512-512 needs one 64-byte output block;
    // the output counter is zero encoded into the first 8 little-endian bytes.
    std::uint8_t output_block[64];
    #pragma unroll
    for (int i = 0; i < 64; ++i) output_block[i] = 0;
    skein512_ubi(chain, output_block, 8, 63, true, true);

    #pragma unroll
    for (int i = 0; i < 8; ++i) skein_store_le64(out + i * 8, chain[i]);
}

} // namespace yerbas::cuda::core
