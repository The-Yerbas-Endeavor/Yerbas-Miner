#pragma once

#include <cstdint>

namespace yerbas::cuda::cryptonight {

// CryptoNight executes AES continuously during scratchpad setup/collapse and
// once per memory-hard loop iteration. Keeping the 256-byte S-box as a local
// constexpr array made every CUDA thread carry an indexable copy that NVCC can
// spill to local memory. Pascal has a broadcast-friendly constant cache, so
// keep one read-only S-box per translation unit in device constant memory.
// Internal linkage avoids cross-TU device-link conflicts for this header-only
// implementation.
static __device__ __constant__ const std::uint8_t kAesSbox[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

__device__ __forceinline__ std::uint8_t aes_sbox(std::uint8_t x)
{
    return kAesSbox[x];
}

__device__ __forceinline__ std::uint8_t xtime(std::uint8_t x)
{
    // Branch-free GF(2^8) multiply-by-2. NVCC normally predicates the old
    // ternary, but this form keeps the hot AES round explicitly branchless.
    const std::uint8_t carry = static_cast<std::uint8_t>(0U - (x >> 7));
    return static_cast<std::uint8_t>((x << 1) ^ (0x1bU & carry));
}

__device__ __forceinline__ void aes_round(std::uint8_t block[16], const std::uint8_t key[16])
{
    std::uint8_t s[16];
    // SubBytes + ShiftRows (AES state is column-major).
    s[0]=aes_sbox(block[0]);  s[1]=aes_sbox(block[5]);  s[2]=aes_sbox(block[10]); s[3]=aes_sbox(block[15]);
    s[4]=aes_sbox(block[4]);  s[5]=aes_sbox(block[9]);  s[6]=aes_sbox(block[14]); s[7]=aes_sbox(block[3]);
    s[8]=aes_sbox(block[8]);  s[9]=aes_sbox(block[13]); s[10]=aes_sbox(block[2]); s[11]=aes_sbox(block[7]);
    s[12]=aes_sbox(block[12]);s[13]=aes_sbox(block[1]); s[14]=aes_sbox(block[6]); s[15]=aes_sbox(block[11]);

    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        const int i = c * 4;
        const std::uint8_t a0=s[i], a1=s[i+1], a2=s[i+2], a3=s[i+3];
        const std::uint8_t t = static_cast<std::uint8_t>(a0 ^ a1 ^ a2 ^ a3);
        block[i]   = static_cast<std::uint8_t>(a0 ^ t ^ xtime(static_cast<std::uint8_t>(a0 ^ a1)) ^ key[i]);
        block[i+1] = static_cast<std::uint8_t>(a1 ^ t ^ xtime(static_cast<std::uint8_t>(a1 ^ a2)) ^ key[i+1]);
        block[i+2] = static_cast<std::uint8_t>(a2 ^ t ^ xtime(static_cast<std::uint8_t>(a2 ^ a3)) ^ key[i+2]);
        block[i+3] = static_cast<std::uint8_t>(a3 ^ t ^ xtime(static_cast<std::uint8_t>(a3 ^ a0)) ^ key[i+3]);
    }
}

__device__ __forceinline__ void aes256_expand_key(const std::uint8_t key[32], std::uint8_t expanded[240])
{
    for (int i = 0; i < 32; ++i) expanded[i] = key[i];
    std::uint8_t rcon = 1;
    int generated = 32;
    while (generated < 240) {
        std::uint8_t t[4] = {expanded[generated-4], expanded[generated-3], expanded[generated-2], expanded[generated-1]};
        if ((generated % 32) == 0) {
            const std::uint8_t x = t[0];
            t[0] = static_cast<std::uint8_t>(aes_sbox(t[1]) ^ rcon);
            t[1] = aes_sbox(t[2]); t[2] = aes_sbox(t[3]); t[3] = aes_sbox(x);
            rcon = xtime(rcon);
        } else if ((generated % 32) == 16) {
            t[0]=aes_sbox(t[0]); t[1]=aes_sbox(t[1]); t[2]=aes_sbox(t[2]); t[3]=aes_sbox(t[3]);
        }
        for (int i = 0; i < 4 && generated < 240; ++i) {
            expanded[generated] = static_cast<std::uint8_t>(expanded[generated - 32] ^ t[i]);
            ++generated;
        }
    }
}

__device__ __forceinline__ void aes_single_round(const std::uint8_t in[16], std::uint8_t out[16], const std::uint8_t round_key[16])
{
    for (int i = 0; i < 16; ++i) out[i] = in[i];
    aes_round(out, round_key);
}

// Matches Yerbas Core aesb_pseudo_round(): ten normal AES rounds, each with
// MixColumns, using the first ten expanded round keys.
__device__ __forceinline__ void aes_pseudo_round(std::uint8_t block[16], const std::uint8_t expanded[240])
{
    #pragma unroll
    for (int round = 0; round < 10; ++round) aes_round(block, expanded + round * 16);
}

} // namespace yerbas::cuda::cryptonight
