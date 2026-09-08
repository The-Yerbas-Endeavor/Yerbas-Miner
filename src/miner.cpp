#include "miner.h"
#include "cpu/cpu_autotune.h"
#include "cpu/cpu_features.h"
#include "cpu/cpu_worker_pool.h"
#include "ghostrider/ghostrider.h"
#include "stratum/stratum.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <streambuf>
#include <thread>
#include <utility>

#ifdef YERBAS_HAS_CUDA
#include "cuda/cuda_backend.h"
#include "cuda/core_coverage.h"
#endif

namespace yerbas {
namespace {
std::atomic_bool g_mining_stop_requested{false};

constexpr auto kDevFeeFirstDelay = std::chrono::minutes(3);
constexpr auto kDevFeeInterval = std::chrono::hours(1);
constexpr auto kDevFeeMiningTime = std::chrono::seconds(60);
constexpr auto kDevFeeSetupLimit = std::chrono::seconds(4);
constexpr auto kDevFeeHashStallLimit = std::chrono::seconds(3);
constexpr const char* kDevPoolUrl = "stratum+tcp://pool.yerbas.org:3333";
constexpr const char* kDevPoolUser = "yYoUt7DosfK6CB4XzLuZSf43auMZFfFFxY";
constexpr const char* kDevPoolWorker = "ymdev";
constexpr const char* kDevPoolPassword = "x";

class NullStreamBuffer final : public std::streambuf {
protected:
    int overflow(int ch) override { return traits_type::not_eof(ch); }
};

class ScopedIostreamSilence {
public:
    ScopedIostreamSilence()
        : cout_buffer_(std::cout.rdbuf(&null_)),
          cerr_buffer_(std::cerr.rdbuf(&null_))
    {
    }

    ~ScopedIostreamSilence() { restore(); }

    void restore()
    {
        if (restored_) return;
        std::cout.rdbuf(cout_buffer_);
        std::cerr.rdbuf(cerr_buffer_);
        restored_ = true;
    }

private:
    NullStreamBuffer null_;
    std::streambuf* cout_buffer_{nullptr};
    std::streambuf* cerr_buffer_{nullptr};
    bool restored_{false};
};

void handle_signal(int)
{
    g_mining_stop_requested.store(true, std::memory_order_relaxed);
    request_stop();
}

bool global_mining_stop_requested()
{
    return g_mining_stop_requested.load(std::memory_order_relaxed) || stop_requested();
}

bool run_user_session_until(stratum::Client& client,
                            std::chrono::steady_clock::time_point deadline)
{
    std::atomic_bool session_stop{false};
    std::atomic_bool session_done{false};
    std::thread runner([&]() {
        (void)client.run(session_stop);
        session_done.store(true, std::memory_order_release);
    });

    while (!session_done.load(std::memory_order_acquire)) {
        if (global_mining_stop_requested() || std::chrono::steady_clock::now() >= deadline) {
            session_stop.store(true, std::memory_order_relaxed);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    if (global_mining_stop_requested())
        session_stop.store(true, std::memory_order_relaxed);

    if (runner.joinable()) runner.join();
    return !global_mining_stop_requested();
}

bool run_developer_fee_round(stratum::Client& client,
                             const PoolConfig& user_pool,
                             const std::string& user_worker)
{
    PoolConfig dev_pool;
    dev_pool.url = kDevPoolUrl;
    dev_pool.user = kDevPoolUser;
    dev_pool.password = kDevPoolPassword;
    client.set_pool_session(dev_pool, kDevPoolWorker);

    std::fprintf(stderr, "[DEV FEE] Starting developer mining round\n");

    std::atomic_bool session_stop{false};
    std::atomic_bool session_done{false};
    bool completed = false;
    bool became_ready = false;

    {
        // Keep the compiled developer identity out of normal console output.
        // Status messages for the fee controller use stdio directly while the
        // Stratum client's iostream output is muted for this short session.
        ScopedIostreamSilence silence;
        std::thread runner([&]() {
            (void)client.run(session_stop);
            session_done.store(true, std::memory_order_release);
        });

        const auto setup_deadline = std::chrono::steady_clock::now() + kDevFeeSetupLimit;
        while (!session_done.load(std::memory_order_acquire) &&
               !client.session_mining_ready() &&
               !global_mining_stop_requested() &&
               std::chrono::steady_clock::now() < setup_deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        if (!global_mining_stop_requested() &&
            !session_done.load(std::memory_order_acquire) &&
            client.session_mining_ready()) {
            became_ready = true;
            std::fprintf(stderr, "[DEV FEE] Developer pool ready\n");
            std::fprintf(stderr, "[DEV FEE] 60-second mining period started\n");

            const auto mining_started = std::chrono::steady_clock::now();
            auto last_progress = mining_started;
            std::uint64_t last_hashes = client.hashes_done_snapshot();

            while (!global_mining_stop_requested() &&
                   !session_done.load(std::memory_order_acquire)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                const auto now = std::chrono::steady_clock::now();
                const std::uint64_t hashes = client.hashes_done_snapshot();
                if (hashes != last_hashes) {
                    last_hashes = hashes;
                    last_progress = now;
                }

                if (now - mining_started >= kDevFeeMiningTime) {
                    completed = true;
                    break;
                }

                // A stopped hash counter means the dev connection/job is no
                // longer producing work. Abort before Client::run reaches its
                // normal five-second reconnect attempt; do not reclaim time.
                if (now - last_progress >= kDevFeeHashStallLimit) break;
            }
        }

        session_stop.store(true, std::memory_order_relaxed);
        if (runner.joinable()) runner.join();
        silence.restore();
    }

    client.set_pool_session(user_pool, user_worker);

    if (global_mining_stop_requested()) return false;

    if (completed) {
        std::cout << "[DEV FEE] Developer mining round complete\n";
    } else if (!became_ready) {
        std::cout << "[DEV FEE] Developer pool unavailable; round skipped\n";
    } else {
        std::cout << "[DEV FEE] Developer mining round interrupted; round skipped\n";
    }
    std::cout << "[DEV FEE] Returning to user pool\n";
    return true;
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

void set_environment_value(const char* name, const char* value)
{
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void set_gpu_tune_environment(const std::string& mode)
{
    // Keep one authoritative user-facing policy while translating it to the
    // legacy/internal CUDA controls used by the individual tuning layers.
    set_environment_value("YERBAS_GPU_TUNE_MODE", mode.c_str());

    if (mode == "full") {
        // Fresh bounded GPU calibration plus fresh CryptoNight selectors,
        // production geometry and stagger policy. Every CUDA cache loader that
        // honors YERBAS_CUDA_RETUNE will be bypassed for this run.
        set_environment_value("YERBAS_GPU_AUTOTUNE", "1");
        set_environment_value("YERBAS_CUDA_RETUNE", "1");
    } else {
        // Config is authoritative for production starts. Avoid inheriting a
        // stale environment setting from a launcher/shell and accidentally
        // turning auto/off into a full retune.
        set_environment_value("YERBAS_GPU_AUTOTUNE", "0");
        set_environment_value("YERBAS_CUDA_RETUNE", "0");
    }
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
    set_gpu_tune_environment(config_.gpu.gpu_tune);

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
    std::cout << "GPU tuning: " << config_.gpu.gpu_tune;
    if (config_.gpu.gpu_tune == "full")
        std::cout << " | bounded calibration=fresh | CUDA selector caches=bypass";
    else if (config_.gpu.gpu_tune == "auto")
        std::cout << " | cache policy=normal";
    else
        std::cout << " | bounded calibration=off";
    std::cout << '\n';
    if (config_.gpu.gpu_tune == "full")
        std::cout << "[AUTOTUNE] GPU phase will run during CUDA initialization\n";

    stratum::Client stratum_client(config_);
    if (stop_requested()) {
        std::cout << "Startup cancelled by user.\n";
        return 130;
    }
    stratum_client.print_connection_plan();
    std::cout << "Developer fee: 1.67% | 60 seconds/hour | first round after 3 minutes\n";

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

    const PoolConfig user_pool = config_.pool;
    const std::string user_worker = config_.miner.worker;
    auto next_dev_round = std::chrono::steady_clock::now() + kDevFeeFirstDelay;

    while (!global_mining_stop_requested()) {
        if (!run_user_session_until(stratum_client, next_dev_round)) break;
        if (global_mining_stop_requested()) break;

        if (!run_developer_fee_round(stratum_client, user_pool, user_worker)) break;
        next_dev_round += kDevFeeInterval;

        // Never stack missed fees. If the machine slept or a round took
        // unusually long, advance to the next future hourly slot.
        const auto now = std::chrono::steady_clock::now();
        while (next_dev_round <= now) next_dev_round += kDevFeeInterval;
    }

    return global_mining_stop_requested() ? 130 : 0;
}

} // namespace yerbas