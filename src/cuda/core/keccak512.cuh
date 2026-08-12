#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace yerbas::cuda::core {

__device__ __forceinline__ std::uint64_t rotl64(std::uint64_t x, int n)
{
    return n == 0 ? x : ((x << n) | (x >> (64 - n)));
}

__device__ __forceinline__ std::uint64_t load64_le(const std::uint8_t* p)
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

__device__ __forceinline__ void store64_le(std::uint8_t* p, std::uint64_t v)
{
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        p[i] = static_cast<std::uint8_t>(v >> (i * 8));
    }
}

__device__ __forceinline__ void keccak_f1600(std::uint64_t s[25])
{
    constexpr std::uint64_t rc[24] = {
        0x0000000000000001ULL, 0x0000000000008082ULL,
        0x800000000000808aULL, 0x8000000080008000ULL,
        0x000000000000808bULL, 0x0000000080000001ULL,
        0x8000000080008081ULL, 0x8000000000008009ULL,
        0x000000000000008aULL, 0x0000000000000088ULL,
        0x0000000080008009ULL, 0x000000008000000aULL,
        0x000000008000808bULL, 0x800000000000008bULL,
        0x8000000000008089ULL, 0x8000000000008003ULL,
        0x8000000000008002ULL, 0x8000000000000080ULL,
        0x000000000000800aULL, 0x800000008000000aULL,
        0x8000000080008081ULL, 0x8000000000008080ULL,
        0x0000000080000001ULL, 0x8000000080008008ULL
    };

    constexpr int rho[25] = {
         0,  1, 62, 28, 27,
        36, 44,  6, 55, 20,
         3, 10, 43, 25, 39,
        41, 45, 15, 21,  8,
        18,  2, 61, 56, 14
    };

    #pragma unroll 1
    for (int round = 0; round < 24; ++round) {
        std::uint64_t c[5];
        std::uint64_t d[5];
        std::uint64_t b[25];

        #pragma unroll
        for (int x = 0; x < 5; ++x) {
            c[x] = s[x] ^ s[x + 5] ^ s[x + 10] ^ s[x + 15] ^ s[x + 20];
        }
        #pragma unroll
        for (int x = 0; x < 5; ++x) {
            d[x] = c[(x + 4) % 5] ^ rotl64(c[(x + 1) % 5], 1);
        }
        #pragma unroll
        for (int y = 0; y < 5; ++y) {
            #pragma unroll
            for (int x = 0; x < 5; ++x) {
                s[x + 5 * y] ^= d[x];
            }
        }

        // Rho + Pi: B[y, (2x+3y) mod 5] = ROT(A[x,y], rho[x,y]).
        #pragma unroll
        for (int y = 0; y < 5; ++y) {
            #pragma unroll
            for (int x = 0; x < 5; ++x) {
                const int nx = y;
                const int ny = (2 * x + 3 * y) % 5;
                b[nx + 5 * ny] = rotl64(s[x + 5 * y], rho[x + 5 * y]);
            }
        }

        #pragma unroll
        for (int y = 0; y < 5; ++y) {
            #pragma unroll
            for (int x = 0; x < 5; ++x) {
                s[x + 5 * y] = b[x + 5 * y] ^
                    ((~b[((x + 1) % 5) + 5 * y]) & b[((x + 2) % 5) + 5 * y]);
            }
        }

        s[0] ^= rc[round];
    }
}

// SPHlib's sph_keccak512 is the original Keccak-512 construction, not the
// later FIPS SHA3-512 domain separation. Rate = 576 bits (72 bytes) and the
// Keccak pad10*1 delimiter starts with 0x01.
__device__ __forceinline__ void keccak512(const std::uint8_t* input,
                                          std::size_t length,
                                          std::uint8_t output[64])
{
    constexpr std::size_t rate = 72;
    std::uint64_t state[25]{};

    while (length >= rate) {
        #pragma unroll
        for (int i = 0; i < 9; ++i) {
            state[i] ^= load64_le(input + i * 8);
        }
        keccak_f1600(state);
        input += rate;
        length -= rate;
    }

    std::uint8_t block[rate]{};
    for (std::size_t i = 0; i < length; ++i) block[i] = input[i];
    block[length] ^= 0x01U;
    block[rate - 1] ^= 0x80U;

    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        state[i] ^= load64_le(block + i * 8);
    }
    keccak_f1600(state);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        store64_le(output + i * 8, state[i]);
    }
}

} // namespace yerbas::cuda::core
