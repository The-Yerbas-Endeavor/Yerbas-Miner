#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "config.h"

#ifdef YERBAS_HAS_CUDA
#include "cuda/cuda_backend.h"
#endif

namespace yerbas::stratum {

struct Endpoint {
    std::string scheme;
    std::string host;
    unsigned short port{0};
};

struct MiningJob {
    std::string job_id;
    std::string prevhash;
    std::string coinb1;
    std::string coinb2;
    std::vector<std::string> merkle_branch;
    std::string version;
    std::string nbits;
    std::string ntime;
    bool clean_jobs{false};
    bool valid{false};
};

Endpoint parse_endpoint(const std::string& url);

class Client {
public:
    explicit Client(const AppConfig& config);

    void print_connection_plan() const;
    bool ready() const noexcept;
    int run(std::atomic_bool& stop_requested);

private:
    bool run_session(std::atomic_bool& stop_requested);
    void handle_message(const std::string& line);
    std::string login_user() const;

    bool build_header(std::array<std::uint8_t, 80>& header,
                      std::string& extranonce2_hex,
                      std::uint32_t nonce) const;
    bool mine_one(std::intptr_t socket_value);
    bool mine_cpu_batch(std::intptr_t socket_value);
#ifdef YERBAS_HAS_CUDA
    bool mine_gpu_batch(std::intptr_t socket_value);
    bool mine_hybrid_round(std::intptr_t socket_value);
    void initialize_gpu_engines();
    void upload_gpu_job();
#endif
    bool submit_share(std::intptr_t socket_value,
                      const std::string& extranonce2_hex,
                      std::uint32_t nonce,
                      const std::string& source);

    void set_target_hex(const std::string& target_hex);
    void set_difficulty(double difficulty);
    void report_stats(bool force = false);

    AppConfig config_;
    Endpoint endpoint_;
    std::uint64_t received_jobs_{0};
    std::uint64_t hashes_done_{0};
    std::uint64_t cpu_hashes_done_{0};
    std::uint64_t shares_submitted_{0};
    std::uint64_t shares_accepted_{0};
    std::uint64_t shares_rejected_{0};
    bool subscribed_{false};
    bool authorized_{false};

    std::string extranonce1_;
    std::size_t extranonce2_size_{4};
    std::uint64_t extranonce2_counter_{0};
    std::uint32_t nonce_{0x80000000U};
    MiningJob job_;
    std::array<std::uint8_t, 32> target_le_{};
    bool target_ready_{false};
    double difficulty_{0.0};

    std::chrono::steady_clock::time_point mining_started_{};
    std::chrono::steady_clock::time_point last_report_{};
    std::uint64_t hashes_at_last_report_{0};
    std::uint64_t cpu_hashes_at_last_report_{0};

#ifdef YERBAS_HAS_CUDA
    struct GpuWorker {
        int device_id{-1};
        std::unique_ptr<cuda::BatchEngine> engine;
        std::uint32_t region_start{0};
        std::uint32_t region_end{0};
        std::uint32_t next_nonce{0};
        std::uint64_t hashes_done{0};
        std::uint64_t hashes_at_last_report{0};
    };
    std::vector<GpuWorker> gpu_workers_;
    bool gpu_job_loaded_{false};
    bool gpu_pipeline_ready_{false};
#endif
};

} // namespace yerbas::stratum
