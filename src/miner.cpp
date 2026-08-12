#include "miner.h"
#include "ghostrider/ghostrider.h"
#include "stratum/stratum.h"

#include <atomic>
#include <csignal>
#include <iostream>
#include <utility>

#ifdef YERBAS_HAS_CUDA
#include "cuda/cuda_backend.h"
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

    std::cout << "Yerbas Miner 0.5.0\n";
    std::cout << "Config: " << config_.config_path << "\n";
    std::cout << "GhostRider reference: "
              << (ghostrider::reference_ready() ? "ready" : "scaffold") << "\n";

    stratum::Client stratum_client(config_);
    stratum_client.print_connection_plan();

#ifdef YERBAS_HAS_CUDA
    if (config_.gpu.enabled) {
        const auto devices = cuda::enumerate_devices();
        std::cout << "CUDA GPUs detected: " << devices.size() << "\n";

        if (devices.empty()) {
            std::cerr << "CUDA status: no compatible NVIDIA GPUs detected\n";
            std::cerr << "GPU mining is enabled, so CPU fallback is disabled.\n";
#ifdef _WIN32
            pause_before_exit();
#endif
            return 3;
        }

        cuda::print_devices();
        if (devices.size() == 1) {
            std::cout << "GPU mode: single GPU\n";
        } else {
            std::cout << "GPU mode: multi-GPU (" << devices.size()
                      << " GPUs available)\n";
            std::cout << "Each GPU can run an independent GhostRider nonce range.\n";
        }

        // Do not silently run the CPU reference miner when the GPU pipeline is
        // incomplete. The CUDA artifact is a GPU miner first; CPU remains only
        // a correctness/reference implementation.
        cuda::BatchEngine readiness_probe(devices.front().id, 1);
        if (!readiness_probe.hash_pipeline_ready()) {
            std::cerr << "CUDA GhostRider pipeline is not complete yet.\n";
            std::cerr << "CPU fallback is disabled because GPU mining is enabled.\n";
            std::cerr << "No mining work will be performed until the full CUDA pipeline is ready.\n";
#ifdef _WIN32
            pause_before_exit();
#endif
            return 4;
        }
    } else {
        std::cout << "CUDA backend: disabled by configuration\n";
    }
#else
    if (config_.gpu.enabled) {
        std::cerr << "CUDA backend: not built in this binary\n";
        std::cerr << "GPU mining is enabled in config.json, so CPU fallback is disabled.\n";
        std::cerr << "Use the yerbas-miner-windows-cuda-x86_64 artifact.\n";
#ifdef _WIN32
        pause_before_exit();
#endif
        return 3;
    }
    std::cout << "CUDA backend: not built; CPU reference mode explicitly selected\n";
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
