#include "miner.h"
#include "ghostrider/ghostrider.h"
#include "stratum/stratum.h"

#include <atomic>
#include <csignal>
#include <iostream>
#include <utility>

#ifdef YERBAS_HAS_CUDA
int yerbas_cuda_device_count();
void yerbas_cuda_print_devices();
#endif

namespace yerbas {
namespace {
std::atomic_bool g_stop_requested{false};

void handle_signal(int)
{
    g_stop_requested.store(true);
}
}

Miner::Miner(AppConfig config)
    : config_(std::move(config))
{
}

int Miner::run()
{
    std::signal(SIGINT, handle_signal);
#ifdef SIGTERM
    std::signal(SIGTERM, handle_signal);
#endif

    std::cout << "Yerbas Miner 0.3.0\n";
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
    std::cout << "CUDA backend: not built in this portable binary\n";
#endif

    std::cout << "Configured GPU device ids:";
    for (const int device : config_.gpu.devices) {
        std::cout << ' ' << device;
    }
    std::cout << "\nGPU intensity: " << config_.gpu.intensity << " (0 = auto)\n";

    if (!stratum_client.ready()) {
        std::cout << "Pool configuration is incomplete. Edit config.json or use --pool and --user.\n";
        std::cout << "Example: yerbas-miner.exe --pool stratum+tcp://pool.example.com:3032 --user YOUR_YERB_ADDRESS\n";
        return 2;
    }

    return stratum_client.run(g_stop_requested);
}

} // namespace yerbas
