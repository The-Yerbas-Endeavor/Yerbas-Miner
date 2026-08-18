#pragma once

#include <cstddef>
#include <cstdint>
#include "jh_device_types.cuh"

namespace yerbas::cuda::core::jh_detail {
#include "jh_device_impl.cuh"
}

namespace yerbas::cuda::core {

__device__ __forceinline__ void jh512(const std::uint8_t* input,
                                      std::size_t length,
                                      std::uint8_t out[64])
{
    sph_jh512_context ctx;
    jh_detail::sph_jh512_init(&ctx);
    jh_detail::sph_jh512(&ctx, input, length);
    jh_detail::sph_jh512_close(&ctx, out);
}

} // namespace yerbas::cuda::core
