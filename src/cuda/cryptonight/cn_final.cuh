#pragma once

// BLAKE-256 uses the exact CryptoNote implementation from pinned Yerbas Core.
// Groestl/JH/Skein use the already validated sphlib device implementations.
// The Core c_groestl.c translation is not byte-for-byte equivalent on CUDA,
// despite receiving the exact matching 200-byte post-Keccak state.
#include "cn_blake256_device.cuh"
#include "groestl_device_types.cuh"
#include "jh_device_types.cuh"
#include "skein_device_types.cuh"

#include <cstdint>

namespace yerbas::cuda::cryptonight::cn_groestl_sph {
#include "groestl_device_impl.cuh"
}
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
    case 1: {
        sph_groestl256_context ctx;
        cn_groestl_sph::sph_groestl256_init(&ctx);
        cn_groestl_sph::sph_groestl256(&ctx, state, 200);
        cn_groestl_sph::sph_groestl256_close(&ctx, out);
        break;
    }
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
