#pragma once

// These files are generated at CMake configure time from the exact pinned
// Yerbas Core cryptonote c_* implementations.
#include "cn_blake256_device.cuh"
#include "cn_groestl_device.cuh"
#include "cn_jh_device.cuh"
#include "cn_skein_device.cuh"

#include <cstdint>

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
        cn_groestl::groestl(state, 200 * 8, out);
        break;
    case 2:
        (void)cn_jh::jh_hash(256, state, 200 * 8, out);
        break;
    default:
        (void)cn_skein::c_skein_hash(256, state, 200 * 8, out);
        break;
    }
}

} // namespace yerbas::cuda::cryptonight
