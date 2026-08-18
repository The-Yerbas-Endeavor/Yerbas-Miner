#pragma once

#include <cstddef>
#include <cstdint>
#include "fugue_device_types.cuh"

namespace yerbas::cuda::core::fugue_detail {
#include "fugue_device_impl.cuh"
}

namespace yerbas::cuda::core {

__device__ __forceinline__ void fugue512(const std::uint8_t* input,
                                         std::size_t length,
                                         std::uint8_t out[64])
{
    sph_fugue512_context ctx;
    fugue_detail::sph_fugue512_init(&ctx);
    fugue_detail::sph_fugue512(&ctx, input, length);
    fugue_detail::sph_fugue512_close(&ctx, out);
}

} // namespace yerbas::cuda::core
