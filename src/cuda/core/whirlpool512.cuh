#pragma once

#include <cstddef>
#include <cstdint>
#include "whirlpool_device_types.cuh"

namespace yerbas::cuda::core::whirlpool_detail {
#include "whirlpool_device_impl.cuh"
}

namespace yerbas::cuda::core {

__device__ __forceinline__ void whirlpool512(const std::uint8_t* input,
                                             std::size_t length,
                                             std::uint8_t out[64])
{
    sph_whirlpool_context ctx;
    whirlpool_detail::sph_whirlpool_init(&ctx);
    whirlpool_detail::sph_whirlpool(&ctx, input, length);
    whirlpool_detail::sph_whirlpool_close(&ctx, out);
}

} // namespace yerbas::cuda::core
