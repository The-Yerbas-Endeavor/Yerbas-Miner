#pragma once

#include <cstddef>
#include <cstdint>
#include "hamsi_device_types.cuh"

namespace yerbas::cuda::core::hamsi_detail {
#include "hamsi_device_impl.cuh"
}

namespace yerbas::cuda::core {

__device__ __forceinline__ void hamsi512(const std::uint8_t* input,
                                         std::size_t length,
                                         std::uint8_t out[64])
{
    sph_hamsi512_context ctx;
    hamsi_detail::sph_hamsi512_init(&ctx);
    hamsi_detail::sph_hamsi512(&ctx, input, length);
    hamsi_detail::sph_hamsi512_close(&ctx, out);
}

} // namespace yerbas::cuda::core
