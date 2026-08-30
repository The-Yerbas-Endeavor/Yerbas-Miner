#include "miner.h"
#include "cpu/cpu_autotune.h"
#include "cpu/cpu_features.h"
#include "cpu/cpu_worker_pool.h"
#include "ghostrider/ghostrider.h"
#include "stratum/stratum.h"

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <utility>

#ifdef YERBAS_HAS_CUDA
#include "cuda/cuda_backend.h"
#include "cuda/core_coverage.h"
#endif

namespace yerbas {
namespace {
std::atomic_bool g_mining_stop_requested{false};

void handle_signal(int)
{
    g_mining_stop_requested.store(true, std::memory_order_relaxed);
    request_stop();
}

void print_cpu_capabilities()
{
    const auto features = cpu::detect_x86_features();
    if (features.available) {
        std::cout << "CPU features:"
                  << " AES=" << cpu::yes_no(features.aes)
                  << " AVX=" << cpu::yes_no(features.avx)
                  << " AVX2=" << cpu::yes_no(features.avx2)
                  << " BMI2=" << cpu::yes_no(features.bmi2)
                  << " SSE4.2=" << cpu::yes_no(features.sse42)
                  << " OSXSAVE=" << cpu::yes_no(features.osxsave)
                  << " YMM=" << cpu::yes_no(features.ymm_state)
#ifdef YERBAS_NATIVE_CPU_BUILD
                  << " | build=native"
#else
                  << " | build=portable+runtime-dispatch"
#endif
                  << '\n';
    } else {
        std::cout << "CPU features: non-x86 runtime probe unavailable"
#ifdef YERBAS_NATIVE_CPU_BUILD
                  << " | build=native"
#else
                  << " | build=portable"
#endif
                  << '\n';
    }
}

#ifdef _WIN32
void pause_before_exit()
{
    std::cout << "\nPress Enter to close..." << std::flush;
    std::cin.get();
}
#endif

void set_gpu_autotune_environment(bool enabled)
{
    if (!enabled) return;
#ifdef _WIN32
    _putenv_s("YERBAS_GPU_AUTOTUNE", "1");
#else
    setenv("YERBAS_GPU_AUTOTUNE", "1", 1);
#endif
}

void set_cpu_retune_environment(bool enabled)
{
    if (!enabled) return;
#ifdef _WIN32
    _putenv_s("YERBAS_CPU_RETUNE", "1");
#else
    setenv("YERBAS_CPU_RETUNE", "1", 1);
#endif
}
}

Miner::Miner(AppConfig config)
    : config_(std::move(config))
{
}

int Miner::run()
{
    g_mining_stop_requested.store(false, std::memory_order_relaxed);
    g_stop_requested.store(false, std::memory_order_relaxed);
    std::signal(SIGINT, handle_signal);
#ifdef SIGTERM
    std::signal(SIGTERM, handle_signal);
#endif

    const unsigned int hw_threads = std::max(1u, std::thread::hardware_concurrency());
    config_.miner.cpu_lanes = 1;

    if (config_.miner.autotune) {
        std::cout << "[AUTOTUNE] combined calibration requested | CPU=fresh | GPU=fresh\n";
        set_cpu_retune_environment(true);
    }

    if (config_.miner.cpu_enabled && !config_.pool.url.empty() && !config_.pool.user.empty()) {
        if (config_.miner.cpu_tune == "off") {
            if (config_.miner.threads == 0) config_.miner.threads = hw_threads;
            std::cout << "[CPU tuning] off | direct startup"
                      << " | hardware_threads=" << hw_threads
                      << " | workers=" << config_.miner.threads
                      << " | batch=" << config_.miner.cpu_batch
                      << " | lanes=1\n";
        } else {
            std::cout << "[AUTOTUNE] CPU phase starting\n";
            const auto tune = cpu::production_autotune(hw_threads,
                                                       config_.miner.threads,
                                                       config_.miner.cpu_batch,
                                                       config_.miner.cpu_tune,
                                                       &g_mining_stop_requested);
            if (tune.interrupted || stop_requested()) {
                std::cout << "[CPU tune] interrupted by user\n";
                return 130;
            }
            config_.miner.threads = tune.threads;
            config_.miner.cpu_lanes = tune.lanes;
            config_.miner.cpu_batch = tune.batch;
            std::cout << "[CPU production policy] selected"
                      << " | workers=" << config_.miner.threads
                      << " | lanes=" << config_.miner.cpu_lanes
                      << " | batch=" << config_.miner.cpu_batch
                      << " | throughput=" << tune.throughput_hps << " H/s"
                      << (tune.from_cache ? " | source=cache" : " | source=fresh")
                      << '\n';
            std::cout << "[AUTOTUNE] CPU phase complete\n";
        }
    }

    cpu::set_runtime_lane_width(config_.miner.cpu_lanes);
    set_gpu_autotune_environment(config_.gpu.autotune);

    std::cout << "Yerbas Miner 0.5.2\n";
    std::cout << "🌿 Proof of Grass | GhostRider mining engine\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << "Config: " << config_.config_path << "\n";
    std::cout << "GhostRider reference: "
              << (ghostrider::reference_ready() ? "ready" : "scaffold") << "\n";

    const unsigned int cpu_threads = config_.miner.threads == 0 ? hw_threads : config_.miner.threads;
    std::cout << "CPU mining: " << (config_.miner.cpu_enabled ? "enabled" : "disabled")
              << " | threads " << cpu_threads
              << " | batch " << config_.miner.cpu_batch << " / thread"
              << " | lanes " << config_.miner.cpu_lanes
              << " | tune " << config_.miner.cpu_tune << "\n";
    print_cpu_capabilities();
    std::cout << "Hybrid scheduler: " << (config_.miner.hybrid ? "enabled" : "disabled") << "\n";
    if (config_.gpu.autotune)
        std::cout << "[AUTOTUNE] GPU phase will run during CUDA initialization\n";

    stratum::Client stratum_client(config_);
    if (stop_requested()) {
        std::cout << "Startup cancelled by user.\n";
        return 130;
    }
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

    return stratum_client.run(g_mining_stop_requested);
}

} // namespace yerbas
