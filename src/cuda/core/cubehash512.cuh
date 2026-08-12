#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace yerbas::cuda::core {

__device__ __forceinline__ std::uint32_t cube_rotl32(std::uint32_t x, int n)
{
    return (x << n) | (x >> (32 - n));
}

__device__ __forceinline__ std::uint32_t cube_load32_le(const std::uint8_t* p)
{
    return static_cast<std::uint32_t>(p[0]) |
           (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
}

__device__ __forceinline__ void cube_store32_le(std::uint8_t* p, std::uint32_t v)
{
    p[0] = static_cast<std::uint8_t>(v);
    p[1] = static_cast<std::uint8_t>(v >> 8);
    p[2] = static_cast<std::uint8_t>(v >> 16);
    p[3] = static_cast<std::uint8_t>(v >> 24);
}

__device__ __forceinline__ void cubehash_round(std::uint32_t x[32])
{
    std::uint32_t y[16];

    #pragma unroll
    for (int i = 0; i < 16; ++i) x[16 + i] += x[i];
    #pragma unroll
    for (int i = 0; i < 16; ++i) y[i ^ 8] = x[i];
    #pragma unroll
    for (int i = 0; i < 16; ++i) x[i] = cube_rotl32(y[i], 7);
    #pragma unroll
    for (int i = 0; i < 16; ++i) x[i] ^= x[16 + i];
    #pragma unroll
    for (int i = 0; i < 16; ++i) y[i ^ 2] = x[16 + i];
    #pragma unroll
    for (int i = 0; i < 16; ++i) x[16 + i] = y[i];

    #pragma unroll
    for (int i = 0; i < 16; ++i) x[16 + i] += x[i];
    #pragma unroll
    for (int i = 0; i < 16; ++i) y[i ^ 4] = x[i];
    #pragma unroll
    for (int i = 0; i < 16; ++i) x[i] = cube_rotl32(y[i], 11);
    #pragma unroll
    for (int i = 0; i < 16; ++i) x[i] ^= x[16 + i];
    #pragma unroll
    for (int i = 0; i < 16; ++i) y[i ^ 1] = x[16 + i];
    #pragma unroll
    for (int i = 0; i < 16; ++i) x[16 + i] = y[i];
}

__device__ __forceinline__ void cubehash_rounds(std::uint32_t x[32], int rounds)
{
    for (int r = 0; r < rounds; ++r) cubehash_round(x);
}

// CubeHash16/32-512, matching SPHlib's sph_cubehash512 parameters:
// 16 rounds per 32-byte message block, 512-bit output.
__device__ __forceinline__ void cubehash512(const std::uint8_t* input,
                                            std::size_t length,
                                            std::uint8_t output[64])
{
    constexpr int rounds = 16;
    constexpr std::size_t block_bytes = 32;
    std::uint32_t x[32]{};

    x[0] = 64;          // output bytes (512 bits)
    x[1] = block_bytes;
    x[2] = rounds;
    cubehash_rounds(x, 10 * rounds);

    while (length >= block_bytes) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) x[i] ^= cube_load32_le(input + i * 4);
        cubehash_rounds(x, rounds);
        input += block_bytes;
        length -= block_bytes;
    }

    std::uint8_t block[block_bytes]{};
    for (std::size_t i = 0; i < length; ++i) block[i] = input[i];
    block[length] = 0x80U;

    #pragma unroll
    for (int i = 0; i < 8; ++i) x[i] ^= cube_load32_le(block + i * 4);
    cubehash_rounds(x, rounds);

    x[31] ^= 1U;
    cubehash_rounds(x, 10 * rounds);

    #pragma unroll
    for (int i = 0; i < 16; ++i) cube_store32_le(output + i * 4, x[i]);
}

} // namespace yerbas::cuda::core
