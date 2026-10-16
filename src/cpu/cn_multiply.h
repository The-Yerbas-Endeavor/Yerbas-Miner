#pragma once

#include <stdint.h>
#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

#if defined(__SIZEOF_INT128__)
#define YERBAS_CN_MULTIPLY_BACKEND "native-wide"
#elif defined(_MSC_VER) && defined(_M_X64)
#define YERBAS_CN_MULTIPLY_BACKEND "msvc-umul128"
#else
#define YERBAS_CN_MULTIPLY_BACKEND "portable-limbs"
#endif

/* Exact unsigned 64x64 -> 128 multiplication. Keep a limb implementation for
 * compilers without a native wide integer or intrinsic. No host-specific
 * compiler flags are required; the AES caller retains its runtime dispatch. */
static inline uint64_t yerbas_cn_mul128_portable(uint64_t a, uint64_t b,
                                                uint64_t* hi)
{
    const uint64_t ah = a >> 32, al = (uint32_t)a;
    const uint64_t bh = b >> 32, bl = (uint32_t)b;
    const uint64_t cross0 = ah * bl, cross1 = al * bh;
    const uint64_t cross = cross0 + cross1;
    const uint64_t low0 = al * bl;
    const uint64_t low = low0 + (cross << 32);
    *hi = ah * bh + (cross >> 32) +
          ((uint64_t)(cross < cross0) << 32) + (uint64_t)(low < low0);
    return low;
}

static inline uint64_t yerbas_cn_mul128(uint64_t a, uint64_t b, uint64_t* hi)
{
#if defined(__SIZEOF_INT128__)
    const __uint128_t product = (__uint128_t)a * b;
    *hi = (uint64_t)(product >> 64);
    return (uint64_t)product;
#elif defined(_MSC_VER) && defined(_M_X64)
    return _umul128(a, b, hi);
#else
    return yerbas_cn_mul128_portable(a, b, hi);
#endif
}
