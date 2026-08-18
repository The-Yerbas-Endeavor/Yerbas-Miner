#pragma once

#include <cstddef>
#include <cstdint>
#include "bmw_device_types.cuh"

namespace yerbas::cuda::core::bmw_detail {
#include "bmw_device_impl.cuh"
}

namespace yerbas::cuda::core {

__device__ __forceinline__ void bmw512(const std::uint8_t* input,
                                       std::size_t length,
                                       std::uint8_t out[64])
{
    sph_bmw512_context ctx;
    bmw_detail::sph_bmw512_init(&ctx);
    bmw_detail::sph_bmw512(&ctx, input, length);
    bmw_detail::sph_bmw512_close(&ctx, out);
}

} // namespace yerbas::cuda::core
