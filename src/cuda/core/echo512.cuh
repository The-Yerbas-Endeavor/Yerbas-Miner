#pragma once

#include <cstddef>
#include <cstdint>
#include "echo_device_types.cuh"

namespace yerbas::cuda::core::echo_detail {
#include "echo_device_impl.cuh"
}

namespace yerbas::cuda::core {

__device__ __forceinline__ void echo512(const std::uint8_t* input,
                                        std::size_t length,
                                        std::uint8_t out[64])
{
    sph_echo512_context ctx;
    echo_detail::sph_echo512_init(&ctx);
    echo_detail::sph_echo512(&ctx, input, length);
    echo_detail::sph_echo512_close(&ctx, out);
}

} // namespace yerbas::cuda::core
