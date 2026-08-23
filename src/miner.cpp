#include "miner.h"
#include "ghostrider/ghostrider.h"
#include "stratum/stratum.h"

#include <atomic>
#include <csignal>
#include <iostream>
#include <thread>
#include <utility>

#ifdef YERBAS_HAS_CUDA
#include "cuda/cuda_backend.h"
#include "cuda/core_coverage.h"
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

    std::cout << "Yerbas Miner 0.5.2\n";
    std::cout << "🌿 Proof of Grass | GhostRider mining engine\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << "Config: " << config_.config_path << "\n";
    std::cout << "GhostRider reference: "
              << (ghostrider::reference_ready() ? "ready" : "scaffold") << "\n";

    const unsigned int hw_threads = std::max(1u, std::thread::hardware_concurrency());
    const unsigned int cpu_threads = config_.miner.threads == 0 ? hw_threads : config_.miner.threads;
    std::cout << "CPU mining: " << (config_.miner.cpu_enabled ? "enabled" : "disabled")
              << " | threads " << cpu_threads << (config_.miner.threads == 0 ? " (auto)" : "") << "\n";
    std::cout << "Hybrid scheduler: " << (config_.miner.hybrid ? "enabled" : "disabled") << "\n";

    stratum::Client stratum_client(config_);
    stratum_client.print_connection_plan();

#ifdef YERBAS_HAS_CUDA
    if (config_.gpu.enabled) {
        const auto devices = cuda::enumerate_devices();
        std::cout << "CUDA GPUs detected: " << devices.size() << "\n";
        if (devices.empty()) {
            std::cerr << "CUDA status: no compatible NVIDIA GPUs detected\n";
            if (!config_.miner.cpu_enabled) {
#ifdef _WIN32
                pause_before_exit();
#endif
                return 3;
            }
            std::cout << "Hybrid mode: continuing with CPU worker(s) only\n";
        } else {
            cuda::print_devices();
            std::cout << "GPU mode: " << (devices.size() == 1 ? "single GPU" : "multi-GPU") << "\n";

            const auto core_count = cuda::implemented_core_count();
            const auto cn_count = cuda::implemented_cryptonight_count();
            std::cout << "CUDA GhostRider coverage: cores " << core_count << "/" << cuda::kCoreCoverage.size()
                      << " | CryptoNight " << cn_count << "/" << cuda::kCryptoNightCoverage.size() << "\n";

            std::cout << "CUDA-ready cores:";
            for (const auto& core : cuda::kCoreCoverage) {
                if (core.implemented) std::cout << ' ' << static_cast<unsigned int>(core.index) << ':' << core.name;
            }
            std::cout << "\nCUDA pending cores:";
            for (const auto& core : cuda::kCoreCoverage) {
                if (!core.implemented) std::cout << ' ' << static_cast<unsigned int>(core.index) << ':' << core.name;
            }
            std::cout << "\nCUDA pending CryptoNight:";
            for (const auto& variant : cuda::kCryptoNightCoverage) {
                if (!variant.implemented) std::cout << ' ' << variant.name;
            }
            std::cout << '\n';

            // The real per-device BatchEngine instances have already been created by
            // the Stratum client and run the CUDA parity/autotune gates at their full
            // production batch size. Do not construct a one-hash BatchEngine here:
            // doing so reruns the complete CryptoNight tuner against batch=1 and can
            // invalidate parity scratch allocations. Coverage is the readiness gate;
            // kernel correctness is enforced by the production engine validation.
            const bool pipeline_ready = cuda::full_ghostrider_cuda_coverage();
            if (config_.gpu.skip_validation)
                std::cout << "CUDA startup validation: skipped by configuration\n";
            else
                std::cout << "CUDA startup validation: production engine parity/autotune complete\n";

            const bool native_ready = cuda::full_ghostrider_cuda_coverage();
            for (const auto& device : devices) {
                std::cout << "[GPU " << device.id << "] CUDA ready | CC "
                          << device.compute_major << '.' << device.compute_minor
                          << " | GhostRider cores " << core_count << '/' << cuda::kCoreCoverage.size()
                          << " | CN " << cn_count << '/' << cuda::kCryptoNightCoverage.size()
                          << " | mining "
                          << (native_ready ? "native" : (pipeline_ready ? "hybrid-bootstrap" : "blocked"))
                          << "\n";
            }

            if (!pipeline_ready) {
                std::cout << "CUDA GhostRider pipeline: partial/validation mode\n";
                if (config_.miner.cpu_enabled) {
                    std::cout << "Hybrid mode: CPU workers remain active while CUDA stages are completed\n";
                } else {
                    std::cerr << "No usable mining backend: CPU disabled and CUDA pipeline incomplete\n";
#ifdef _WIN32
                    pause_before_exit();
#endif
                    return 4;
                }
            } else if (!native_ready) {
                std::cout << "CUDA GhostRider pipeline: hybrid-bootstrap mining enabled\n";
                std::cout << "Validated CUDA cores run on GPU; pending stages use the pinned CPU reference fallback\n";
            } else {
                std::cout << "CUDA GhostRider pipeline: full native CUDA mining enabled\n";
                std::cout << "🌿 Proof of Grass: growing on CUDA\n";
            }
        }
    } else {
        std::cout << "CUDA backend: disabled by configuration\n";
    }
#else
    if (config_.gpu.enabled) {
        std::cout << "CUDA backend: not built in this binary\n";
        std::cout << "Use yerbas-miner-windows-cuda-x86_64 for GPU + CPU hybrid mining\n";
    }
    if (!config_.miner.cpu_enabled) {
        std::cerr << "No mining backend enabled\n";
#ifdef _WIN32
        pause_before_exit();
#endif
        return 3;
    }
#endif

    if (config_.gpu.devices.empty()) {
        std::cout << "Configured GPU device ids: all detected GPUs\n";
    } else {
        std::cout << "Configured GPU device ids:";
        for (const int device : config_.gpu.devices) std::cout << ' ' << device;
        std::cout << '\n';
    }
    std::cout << "GPU intensity: " << config_.gpu.intensity << " (0 = auto)\n";

    if (!stratum_client.ready()) {
        std::cout << "\nPool configuration is incomplete.\n";
        std::cout << "Edit config.json and set pool.url and pool.user.\n";
#ifdef _WIN32
        pause_before_exit();
#endif
        return 2;
    }

    return stratum_client.run(g_stop_requested);
}

} // namespace yerbas
