#pragma once

#include <cstdint>

// CUDA validation only needs the pinned Yerbas Core fast-hash entry point.
// Keep this lightweight test shim local so the validator does not depend on
// the fetched Core tree exposing cryptonote/slow-hash.h through a particular
// include search path.
namespace crypto {
extern "C" {
void cn_fast_hash(const char* input, char* output, std::uint32_t len);
}

inline void cryptonight_dark_fast_hash(const char* input, char* output, std::uint32_t len)
{
    cn_fast_hash(input, output, len);
}
} // namespace crypto
