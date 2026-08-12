#include "miner.h"
#include "ghostrider/ghostrider.h"

#include <iostream>

#ifdef YERBAS_HAS_CUDA
int yerbas_cuda_device_count();
void yerbas_cuda_print_devices();
#endif

namespace yerbas {

int Miner::run()
{
    std::cout << "Yerbas Miner 0.1.0\n";
    std::cout << "GhostRider reference: "
              << (ghostrider::reference_ready() ? "ready" : "scaffold") << "\n";

#ifdef YERBAS_HAS_CUDA
    const int devices = yerbas_cuda_device_count();
    std::cout << "CUDA devices: " << devices << "\n";
    if (devices > 0) {
        yerbas_cuda_print_devices();
    }
#else
    std::cout << "CUDA backend: not built\n";
#endif

    std::cout << "Mining loop is intentionally disabled until the CPU reference "
                 "hash and test vectors are verified.\n";
    return 0;
}

} // namespace yerbas
