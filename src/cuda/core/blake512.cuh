#pragma once

#include <cstddef>
#include <cstdint>

namespace yerbas::cuda::core {

__device__ __forceinline__ std::uint64_t rotr64(std::uint64_t x, unsigned n)
{
    return (x >> n) | (x << (64U - n));
}

__device__ __forceinline__ std::uint64_t load_be64(const std::uint8_t* p)
{
    return (static_cast<std::uint64_t>(p[0]) << 56) |
           (static_cast<std::uint64_t>(p[1]) << 48) |
           (static_cast<std::uint64_t>(p[2]) << 40) |
           (static_cast<std::uint64_t>(p[3]) << 32) |
           (static_cast<std::uint64_t>(p[4]) << 24) |
           (static_cast<std::uint64_t>(p[5]) << 16) |
           (static_cast<std::uint64_t>(p[6]) << 8)  |
           static_cast<std::uint64_t>(p[7]);
}

__device__ __forceinline__ void store_be64(std::uint8_t* p, std::uint64_t x)
{
    p[0] = static_cast<std::uint8_t>(x >> 56);
    p[1] = static_cast<std::uint8_t>(x >> 48);
    p[2] = static_cast<std::uint8_t>(x >> 40);
    p[3] = static_cast<std::uint8_t>(x >> 32);
    p[4] = static_cast<std::uint8_t>(x >> 24);
    p[5] = static_cast<std::uint8_t>(x >> 16);
    p[6] = static_cast<std::uint8_t>(x >> 8);
    p[7] = static_cast<std::uint8_t>(x);
}

__device__ __forceinline__ void blake512_g(std::uint64_t v[16],
                                          const std::uint64_t m[16],
                                          const std::uint64_t c[16],
                                          int a, int b, int cc, int d,
                                          int x, int y)
{
    v[a] = v[a] + v[b] + (m[x] ^ c[y]);
    v[d] = rotr64(v[d] ^ v[a], 32);
    v[cc] += v[d];
    v[b] = rotr64(v[b] ^ v[cc], 25);
    v[a] = v[a] + v[b] + (m[y] ^ c[x]);
    v[d] = rotr64(v[d] ^ v[a], 16);
    v[cc] += v[d];
    v[b] = rotr64(v[b] ^ v[cc], 11);
}

// BLAKE-512 as used by sphlib/Yerbas Core. GhostRider calls this primitive
// only with an 80-byte header (stage 0) or a 64-byte intermediate state, so
// both inputs fit in one padded 1024-bit block.
__device__ __forceinline__ void blake512(const std::uint8_t* input,
                                         std::size_t length,
                                         std::uint8_t out[64])
{
    const std::uint64_t iv[8] = {
        0x6A09E667F3BCC908ULL, 0xBB67AE8584CAA73BULL,
        0x3C6EF372FE94F82BULL, 0xA54FF53A5F1D36F1ULL,
        0x510E527FADE682D1ULL, 0x9B05688C2B3E6C1FULL,
        0x1F83D9ABFB41BD6BULL, 0x5BE0CD19137E2179ULL
    };
    const std::uint64_t c[16] = {
        0x243F6A8885A308D3ULL, 0x13198A2E03707344ULL,
        0xA4093822299F31D0ULL, 0x082EFA98EC4E6C89ULL,
        0x452821E638D01377ULL, 0xBE5466CF34E90C6CULL,
        0xC0AC29B7C97C50DDULL, 0x3F84D5B5B5470917ULL,
        0x9216D5D98979FB1BULL, 0xD1310BA698DFB5ACULL,
        0x2FFD72DBD01ADFB7ULL, 0xB8E1AFED6A267E96ULL,
        0xBA7C9045F12C7F99ULL, 0x24A19947B3916CF7ULL,
        0x0801F2E2858EFC16ULL, 0x636920D871574E69ULL
    };
    const unsigned char sigma[16][16] = {
        {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
        {14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3},
        {11,8,12,0,5,2,15,13,10,14,3,6,7,1,9,4},
        {7,9,3,1,13,12,11,14,2,6,5,10,4,0,15,8},
        {9,0,5,7,2,4,10,15,14,1,11,12,6,8,3,13},
        {2,12,6,10,0,11,8,3,4,13,7,5,15,14,1,9},
        {12,5,1,15,14,13,4,10,0,7,6,3,9,2,8,11},
        {13,11,7,14,12,1,3,9,5,0,15,4,8,6,2,10},
        {6,15,14,9,11,3,0,8,12,2,13,7,1,4,10,5},
        {10,2,8,4,7,6,1,5,15,11,9,14,3,12,13,0},
        {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
        {14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3},
        {11,8,12,0,5,2,15,13,10,14,3,6,7,1,9,4},
        {7,9,3,1,13,12,11,14,2,6,5,10,4,0,15,8},
        {9,0,5,7,2,4,10,15,14,1,11,12,6,8,3,13},
        {2,12,6,10,0,11,8,3,4,13,7,5,15,14,1,9}
    };

    std::uint8_t block[128];
    #pragma unroll
    for (int i = 0; i < 128; ++i) block[i] = 0;
    for (std::size_t i = 0; i < length; ++i) block[i] = input[i];

    // BLAKE-512 padding: 1 bit, zero fill, 1-bit domain marker for 512-bit
    // output, then the 128-bit big-endian message length. For these inputs the
    // high 64 bits of the length are zero.
    block[length] = 0x80;
    block[111] |= 0x01;
    const std::uint64_t bit_length = static_cast<std::uint64_t>(length) * 8ULL;
    store_be64(block + 112, 0ULL);
    store_be64(block + 120, bit_length);

    std::uint64_t m[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = load_be64(block + i * 8);

    std::uint64_t h[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) h[i] = iv[i];

    std::uint64_t v[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = h[i];
    v[8]  = c[0]; v[9]  = c[1]; v[10] = c[2]; v[11] = c[3];
    v[12] = bit_length ^ c[4];
    v[13] = bit_length ^ c[5];
    v[14] = c[6];
    v[15] = c[7];

    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        const unsigned char* s = sigma[r];
        blake512_g(v,m,c,0,4,8,12,s[0],s[1]);
        blake512_g(v,m,c,1,5,9,13,s[2],s[3]);
        blake512_g(v,m,c,2,6,10,14,s[4],s[5]);
        blake512_g(v,m,c,3,7,11,15,s[6],s[7]);
        blake512_g(v,m,c,0,5,10,15,s[8],s[9]);
        blake512_g(v,m,c,1,6,11,12,s[10],s[11]);
        blake512_g(v,m,c,2,7,8,13,s[12],s[13]);
        blake512_g(v,m,c,3,4,9,14,s[14],s[15]);
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) h[i] ^= v[i] ^ v[i + 8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) store_be64(out + i * 8, h[i]);
}

} // namespace yerbas::cuda::core
