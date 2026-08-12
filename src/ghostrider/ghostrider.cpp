#include "ghostrider/ghostrider.h"

#include <stdexcept>

namespace yerbas::ghostrider {

Hash256 hash_reference(const Work& work)
{
    if (work.data == nullptr || work.size == 0) {
        throw std::invalid_argument("GhostRider work buffer is empty");
    }

    // Deliberately do not return a fake PoW hash. The next implementation step
    // is to import Yerbas Core-compatible GhostRider code and known vectors.
    throw std::runtime_error("GhostRider reference hash is not implemented yet");
}

bool reference_ready() noexcept
{
    return false;
}

} // namespace yerbas::ghostrider
