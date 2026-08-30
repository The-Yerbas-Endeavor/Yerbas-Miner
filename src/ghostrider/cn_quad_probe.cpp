#include "ghostrider/ghostrider.h"
#include "cpu/cn_2way.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace yerbas::ghostrider {
namespace {

struct CnParams {
    std::uint32_t page_size;
    std::uint32_t iterations;
    std::size_t aes_rounds;
};

constexpr std::array<CnParams, 6> kProbeCnParams{{
    {524288U, 131072U, 32768U},
    {524288U, 131072U, 16384U},
    {2097152U, 262144U, 131072U},
    {1048576U, 262144U, 65536U},
    {262144U, 65536U, 16384U},
    {262144U, 65536U, 8192U},
}};

} // namespace

bool optimized_cn_quad_stage(const Hash512& input0,
                             const Hash512& input1,
                             const Hash512& input2,
                             const Hash512& input3,
                             std::uint8_t variant,
                             Hash512& output0,
                             Hash512& output1,
                             Hash512& output2,
                             Hash512& output3) noexcept
{
    if (variant >= kProbeCnParams.size()) return false;
    const auto& params = kProbeCnParams[static_cast<std::size_t>(variant)];
    output0.fill(0);
    output1.fill(0);
    output2.fill(0);
    output3.fill(0);
    return yerbas_cn_hash_quad_4way(
               reinterpret_cast<const char*>(input0.data()),
               reinterpret_cast<const char*>(input1.data()),
               reinterpret_cast<const char*>(input2.data()),
               reinterpret_cast<const char*>(input3.data()),
               reinterpret_cast<char*>(output0.data()),
               reinterpret_cast<char*>(output1.data()),
               reinterpret_cast<char*>(output2.data()),
               reinterpret_cast<char*>(output3.data()),
               64U,
               1,
               params.page_size,
               params.iterations,
               params.aes_rounds) != 0;
}

} // namespace yerbas::ghostrider
