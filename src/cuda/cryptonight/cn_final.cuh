#pragma once

// BLAKE-256 and Groestl-256 use the exact CryptoNote implementations from
// pinned Yerbas Core. JH/Skein temporarily retain the already generated
// sphlib device implementations until their Core translations are made
// self-contained as well.
#include "cn_blake256_device.cuh"
#include "cn_groestl_device.cuh"
#include "jh_device_types.cuh"
#include "skein_device_types.cuh"

#include <cstdint>

namespace yerbas::cuda::cryptonight::cn_jh_sph {
#include "jh_device_impl.cuh"
}
namespace yerbas::cuda::cryptonight::cn_skein_sph {
#include "skein_device_impl.cuh"
}

namespace yerbas::cuda::cryptonight {

__device__ __forceinline__ void dispatch_extra_hash(std::uint8_t selector,
                                                     const std::uint8_t state[200],
                                                     std::uint8_t out[32])
{
    switch (selector & 3U) {
    case 0:
        cn_blake256::blake256_hash(out, state, 200);
        break;
    case 1:
        // Core slow-hash.c: groestl(input, len * 8, output).
        cn_groestl::groestl(state, 200ULL * 8ULL, out);
        break;
    case 2: {
        sph_jh256_context ctx;
        cn_jh_sph::sph_jh256_init(&ctx);
        cn_jh_sph::sph_jh256(&ctx, state, 200);
        cn_jh_sph::sph_jh256_close(&ctx, out);
        break;
    }
    default: {
        sph_skein256_context ctx;
        cn_skein_sph::sph_skein256_init(&ctx);
        cn_skein_sph::sph_skein256(&ctx, state, 200);
        cn_skein_sph::sph_skein256_close(&ctx, out);
        break;
    }
    }
}

} // namespace yerbas::cuda::cryptonight
