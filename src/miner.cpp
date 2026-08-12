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

#ifdef _WIN32
void pause_before_exit()
{
    std::cout << "\nPress Enter to close..." << std::flush;
    std::cin.get();
}
#endif
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

    std::cout << "Yerbas Miner 0.4.1\n";
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
        std::cout << "\nPool configuration is incomplete.\n";
        std::cout << "Edit config.json in the same folder as yerbas-miner.exe and set:\n"
                     "  pool.url\n"
                     "  pool.user\n\n";
        std::cout << "Or run:\n"
                     "  yerbas-miner.exe --pool stratum+tcp://POOL:PORT --user YOUR_YERB_ADDRESS\n";
#ifdef _WIN32
        pause_before_exit();
#endif
        return 2;
    }

    return stratum_client.run(g_stop_requested);
}

} // namespace yerbas
