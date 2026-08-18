#pragma once

#include <cstddef>
#include <cstdint>
#include "shabal_device_types.cuh"

namespace yerbas::cuda::core::shabal_detail {
#include "shabal_device_impl.cuh"
}

namespace yerbas::cuda::core {

__device__ __forceinline__ void shabal512(const std::uint8_t* input,
                                          std::size_t length,
                                          std::uint8_t out[64])
{
    sph_shabal512_context ctx;
    shabal_detail::sph_shabal512_init(&ctx);
    shabal_detail::sph_shabal512(&ctx, input, length);
    shabal_detail::sph_shabal512_close(&ctx, out);
}

} // namespace yerbas::cuda::core
