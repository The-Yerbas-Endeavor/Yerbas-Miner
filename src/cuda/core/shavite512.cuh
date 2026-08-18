#pragma once

#include <cstddef>
#include <cstdint>
#include "shavite_device_types.cuh"

namespace yerbas::cuda::core::shavite_detail {
#include "shavite_device_impl.cuh"
}

namespace yerbas::cuda::core {

__device__ __forceinline__ void shavite512(const std::uint8_t* input,
                                           std::size_t length,
                                           std::uint8_t out[64])
{
    sph_shavite512_context ctx;
    shavite_detail::sph_shavite512_init(&ctx);
    shavite_detail::sph_shavite512(&ctx, input, length);
    shavite_detail::sph_shavite512_close(&ctx, out);
}

} // namespace yerbas::cuda::core
