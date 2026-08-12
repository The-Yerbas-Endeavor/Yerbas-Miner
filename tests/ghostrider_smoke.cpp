#include "ghostrider/ghostrider.h"

#include <array>
#include <cstdint>
#include <iostream>

int main()
{
    std::array<std::uint8_t, 80> header{};
    const yerbas::ghostrider::Work work{header.data(), header.size()};

    if (!yerbas::ghostrider::reference_ready()) {
        std::cerr << "GhostRider reference implementation is not ready\n";
        return 1;
    }

    const auto hash = yerbas::ghostrider::hash_reference(work);
    (void)hash;
    std::cout << "Yerbas Core GhostRider reference smoke test passed\n";
    return 0;
}
