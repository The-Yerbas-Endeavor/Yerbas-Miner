#pragma once

#include <cstddef>
#include <cstdint>

namespace yerbas::cuda::cryptonight {

__device__ __forceinline__ std::uint64_t rotl64(std::uint64_t x, int n)
{
    return (x << n) | (x >> (64 - n));
}

__device__ __forceinline__ void keccakf(std::uint64_t st[25])
{
    constexpr std::uint64_t rndc[24] = {
        0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
        0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
        0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
        0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
        0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
        0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
        0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
        0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
    };
    constexpr int rotc[24] = {
        1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
        27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
    };
    constexpr int piln[24] = {
        10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
        15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
    };

    for (int round = 0; round < 24; ++round) {
        std::uint64_t bc[5];
        for (int i = 0; i < 5; ++i)
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
        for (int i = 0; i < 5; ++i) {
            const std::uint64_t t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5) st[j + i] ^= t;
        }

        std::uint64_t t = st[1];
        for (int i = 0; i < 24; ++i) {
            const int j = piln[i];
            const std::uint64_t saved = st[j];
            st[j] = rotl64(t, rotc[i]);
            t = saved;
        }

        for (int j = 0; j < 25; j += 5) {
            for (int i = 0; i < 5; ++i) bc[i] = st[j + i];
            for (int i = 0; i < 5; ++i)
                st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }
        st[0] ^= rndc[round];
    }
}

__device__ __forceinline__ std::uint64_t load64le(const std::uint8_t* p)
{
    return (std::uint64_t)p[0] | ((std::uint64_t)p[1] << 8) |
           ((std::uint64_t)p[2] << 16) | ((std::uint64_t)p[3] << 24) |
           ((std::uint64_t)p[4] << 32) | ((std::uint64_t)p[5] << 40) |
           ((std::uint64_t)p[6] << 48) | ((std::uint64_t)p[7] << 56);
}

// CryptoNight hash_process(): Keccak-1600 with a 200-byte output state.
__device__ __forceinline__ void keccak1600(const std::uint8_t* input,
                                           std::size_t length,
                                           std::uint8_t state_bytes[200])
{
    std::uint64_t st[25]{};
    constexpr std::size_t rate = 136; // HASH_DATA_AREA from Yerbas Core

    while (length >= rate) {
        for (int i = 0; i < 17; ++i) st[i] ^= load64le(input + i * 8);
        keccakf(st);
        input += rate;
        length -= rate;
    }

    std::uint8_t tail[rate]{};
    for (std::size_t i = 0; i < length; ++i) tail[i] = input[i];
    tail[length] = 0x01;
    tail[rate - 1] |= 0x80;
    for (int i = 0; i < 17; ++i) st[i] ^= load64le(tail + i * 8);
    keccakf(st);

    for (int i = 0; i < 25; ++i) {
        const std::uint64_t v = st[i];
        for (int b = 0; b < 8; ++b)
            state_bytes[i * 8 + b] = static_cast<std::uint8_t>(v >> (8 * b));
    }
}

__device__ __forceinline__ void keccak_permute_state(std::uint8_t state_bytes[200])
{
    std::uint64_t st[25];
    for (int i = 0; i < 25; ++i) st[i] = load64le(state_bytes + i * 8);
    keccakf(st);
    for (int i = 0; i < 25; ++i) {
        const std::uint64_t v = st[i];
        for (int b = 0; b < 8; ++b)
            state_bytes[i * 8 + b] = static_cast<std::uint8_t>(v >> (8 * b));
    }
}

} // namespace yerbas::cuda::cryptonight
