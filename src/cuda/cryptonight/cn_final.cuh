#pragma once

// BLAKE-256 is generated directly from pinned Yerbas Core c_blake256.c.
// Groestl-256, JH-256 and Skein-256 reuse the already generated sphlib device
// implementations from the same pinned Yerbas Core revision. This avoids
// dragging the cryptonote headers' host-only include graph into CUDA while
// preserving the same standardized 256-bit primitives used by cn_slow_hash.
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
