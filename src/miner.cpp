#include "miner.h"
#include "ghostrider/ghostrider.h"
#include "stratum/stratum.h"

#include <iostream>
#include <utility>

#ifdef YERBAS_HAS_CUDA
int yerbas_cuda_device_count();
void yerbas_cuda_print_devices();
#endif

namespace yerbas {

Miner::Miner(AppConfig config)
    : config_(std::move(config))
{
}

int Miner::run()
{
    std::cout << "Yerbas Miner 0.2.0\n";
    std::cout << "Config: " << config_.config_path << "\n";
    std::cout << "GhostRider reference: "
              << (ghostrider::reference_ready() ? "ready" : "scaffold") << "\n";

    stratum::Client stratum_client(config_);
    stratum_client.print_connection_plan();

#ifdef YERBAS_HAS_CUDA
    if (config_.gpu.enabled) {
        const int devices = yerbas_cuda_device_count();
        std::cout << "CUDA devices detected: " << devices << "\n";
        if (devices > 0) {
            yerbas_cuda_print_devices();
        }
    } else {
        std::cout << "CUDA backend: disabled by configuration\n";
    }
#else
    std::cout << "CUDA backend: not built\n";
#endif

    std::cout << "Configured GPU device ids:";
    for (const int device : config_.gpu.devices) {
        std::cout << ' ' << device;
    }
    std::cout << "\nGPU intensity: " << config_.gpu.intensity << " (0 = auto)\n";

    if (!stratum_client.ready()) {
        std::cout << "Mining is not started: configure pool.url and pool.user first.\n";
        return 0;
    }

    std::cout << "Stratum socket/session and share submission are the next implementation stage; "
                 "no shares are submitted by this scaffold yet.\n";
    return 0;
}

} // namespace yerbas
