#include "ghostrider/ghostrider.h"

#include <cstdint>
#include <iostream>

int main()
{
    const std::uint8_t sample[] = {0x59, 0x45, 0x52, 0x42};
    const yerbas::ghostrider::Work work{sample, sizeof(sample)};

    if (yerbas::ghostrider::reference_ready()) {
        (void)yerbas::ghostrider::hash_reference(work);
        std::cout << "GhostRider reference smoke test passed\n";
    } else {
        std::cout << "GhostRider reference not implemented; scaffold is healthy\n";
    }

    return 0;
}
