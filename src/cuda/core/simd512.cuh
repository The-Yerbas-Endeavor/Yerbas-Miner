#pragma once

#include <cstddef>
#include <cstdint>
#include "simd_device_types.cuh"

namespace yerbas::cuda::core::simd_detail {
#include "simd_device_impl.cuh"
}

namespace yerbas::cuda::core {

__device__ __forceinline__ void simd512(const std::uint8_t* input,
                                        std::size_t length,
                                        std::uint8_t out[64])
{
    sph_simd512_context ctx;
    simd_detail::sph_simd512_init(&ctx);
    simd_detail::sph_simd512(&ctx, input, length);
    simd_detail::sph_simd512_close(&ctx, out);
}

} // namespace yerbas::cuda::core
