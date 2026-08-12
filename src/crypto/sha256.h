#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace yerbas::crypto {

using Hash256 = std::array<std::uint8_t, 32>;

Hash256 sha256(const std::uint8_t* data, std::size_t size);
Hash256 double_sha256(const std::uint8_t* data, std::size_t size);
Hash256 double_sha256(const std::vector<std::uint8_t>& data);

} // namespace yerbas::crypto
