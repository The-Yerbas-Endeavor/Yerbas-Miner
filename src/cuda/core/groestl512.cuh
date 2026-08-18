#pragma once

#include <cstddef>
#include <cstdint>
#include "groestl_device_types.cuh"

namespace yerbas::cuda::core::groestl_detail {
#include "groestl_device_impl.cuh"
}

namespace yerbas::cuda::core {

__device__ __forceinline__ void groestl512(const std::uint8_t* input,
                                           std::size_t length,
                                           std::uint8_t out[64])
{
    sph_groestl512_context ctx;
    groestl_detail::sph_groestl512_init(&ctx);
    groestl_detail::sph_groestl512(&ctx, input, length);
    groestl_detail::sph_groestl512_close(&ctx, out);
}

} // namespace yerbas::cuda::core
