#include "ghostrider/ghostrider.h"
#include "ghostrider_vectors.h"

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>

int main()
{
    const auto& header = yerbas::test_vectors::MAINNET_GENESIS_HEADER;
    const yerbas::ghostrider::Work work{header.data(), header.size()};

    if (!yerbas::ghostrider::reference_ready()) {
        std::cerr << "GhostRider reference implementation is not ready\n";
        return 1;
    }

    const auto hash = yerbas::ghostrider::hash_reference(work);
    const bool nonzero = std::any_of(hash.begin(), hash.end(), [](std::uint8_t b) { return b != 0; });
    if (!nonzero) {
        std::cerr << "GhostRider returned an all-zero hash for the real Yerbas genesis header\n";
        return 1;
    }

    std::cout << "Yerbas mainnet genesis header: "
              << yerbas::test_vectors::MAINNET_GENESIS_HEADER_HEX << '\n';
    std::cout << "Yerbas GetHash(): "
              << yerbas::test_vectors::MAINNET_GENESIS_BLOCK_HASH << '\n';
    std::cout << "Yerbas GhostRider PoW hash (internal byte order): ";
    for (const auto byte : hash) {
        std::cout << std::hex << std::setfill('0') << std::setw(2)
                  << static_cast<unsigned int>(byte);
    }
    std::cout << std::dec << '\n';
    std::cout << "Real Yerbas block-header GhostRider smoke test passed\n";
    return 0;
}
