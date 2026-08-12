#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace yerbas::ghostrider {

using Hash256 = std::array<std::uint8_t, 32>;

struct Work {
    const std::uint8_t* data{nullptr};
    std::size_t size{0};
};

// CPU reference implementation wired to the exact GhostRider primitives and
// hash-selection logic from Yerbas Core. For normal mining input, data is the
// serialized 80-byte Yerbas block header (nVersion through nNonce).
Hash256 hash_reference(const Work& work);

bool reference_ready() noexcept;

} // namespace yerbas::ghostrider
