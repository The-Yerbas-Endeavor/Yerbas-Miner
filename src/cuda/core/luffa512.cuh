#pragma once

#include <cstddef>
#include <cstdint>
#include "luffa_device_types.cuh"

namespace yerbas::cuda::core::luffa_detail {
#include "luffa_device_impl.cuh"
}

namespace yerbas::cuda::core {

__device__ __forceinline__ void luffa512(const std::uint8_t* input,
                                         std::size_t length,
                                         std::uint8_t out[64])
{
    sph_luffa512_context ctx;
    luffa_detail::sph_luffa512_init(&ctx);
    luffa_detail::sph_luffa512(&ctx, input, length);
    luffa_detail::sph_luffa512_close(&ctx, out);
}

} // namespace yerbas::cuda::core
