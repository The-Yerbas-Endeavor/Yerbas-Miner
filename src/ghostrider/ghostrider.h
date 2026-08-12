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

// CPU reference entry point. This intentionally remains a placeholder until
// Yerbas Core's exact GhostRider implementation/test vectors are imported.
Hash256 hash_reference(const Work& work);

// Returns true only when the real GhostRider reference implementation has
// replaced the scaffold implementation.
bool reference_ready() noexcept;

} // namespace yerbas::ghostrider
