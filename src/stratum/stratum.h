#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "config.h"
#include "cpu/cpu_worker_pool.h"

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

    MiningJob() = default;
    MiningJob(const MiningJob&) = default;

    MiningJob& operator=(const MiningJob& other)
    {
        if (this == &other) return *this;
        job_id = other.job_id;
        prevhash = other.prevhash;
        coinb1 = other.coinb1;
        coinb2 = other.coinb2;
        merkle_branch = other.merkle_branch;
        version = other.version;
        nbits = other.nbits;
        ntime = other.ntime;
        clean_jobs = other.clean_jobs;
        valid = other.valid;
        ++generation_;
        return *this;
    }

    MiningJob& operator=(MiningJob&& other) noexcept
    {
        if (this == &other) return *this;
        job_id = std::move(other.job_id);
        prevhash = std::move(other.prevhash);
        coinb1 = std::move(other.coinb1);
        coinb2 = std::move(other.coinb2);
        merkle_branch = std::move(other.merkle_branch);
        version = std::move(other.version);
        nbits = std::move(other.nbits);
        ntime = std::move(other.ntime);
        clean_jobs = other.clean_jobs;
        valid = other.valid;
        ++generation_;
        return *this;
    }

    static std::uint64_t generation() noexcept { return generation_.load(std::memory_order_relaxed); }

private:
    inline static std::atomic<std::uint64_t> generation_{0};
};

#ifdef YERBAS_HAS_CUDA
// mining.set_difficulty / mining.set_target can arrive while the current job is
// still active. Re-uploading that same job used to reset every GPU nonce cursor
// (and the CPU cursor through upload_gpu_job), causing duplicate submissions.
// A real mining.notify assignment advances MiningJob::generation(), so a false
// assignment is honored for a new job but ignored for target-only updates to the
// same job. This preserves nonce progress without changing Stratum wire behavior.
class JobLoadedFlag {
public:
    JobLoadedFlag() noexcept : generation_(MiningJob::generation()) {}

    JobLoadedFlag& operator=(bool value) noexcept
    {
        const std::uint64_t current_generation = MiningJob::generation();
        if (value) {
            value_ = true;
            generation_ = current_generation;
        } else if (!value_ || generation_ != current_generation) {
            value_ = false;
            generation_ = current_generation;
        }
        return *this;
    }

    operator bool() const noexcept { return value_; }
    bool operator!() const noexcept { return !value_; }

private:
    bool value_{false};
    std::uint64_t generation_{0};
};
#endif

Endpoint parse_endpoint(const std::string& url);

class Client {
public:
    explicit Client(const AppConfig& config);
    ~Client();

    void print_connection_plan() const;
    bool ready() const noexcept;
    int run(std::atomic_bool& stop_requested);

private:
    bool run_session(std::atomic_bool& stop_requested);
    bool pump_socket_messages(std::intptr_t socket_value, int wait_ms = 0);
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
    void activate_pending_target();
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

    // target_le_/difficulty_ describe the active mining job only. Pool vardiff
    // messages are staged in pending_* and promoted atomically by mining.notify.
    std::array<std::uint8_t, 32> target_le_{};
    bool target_ready_{false};
    double difficulty_{0.0};
    std::array<std::uint8_t, 32> pending_target_le_{};
    bool pending_target_ready_{false};
    double pending_difficulty_{0.0};
    bool pending_difficulty_ready_{false};

    std::string socket_pending_;

    std::chrono::steady_clock::time_point mining_started_{};
    std::chrono::steady_clock::time_point last_report_{};
    std::uint64_t hashes_at_last_report_{0};
    std::uint64_t cpu_hashes_at_last_report_{0};

#ifdef YERBAS_HAS_CUDA
    struct GpuExecutor;
    struct GpuWorker {
        int device_id{-1};
        std::unique_ptr<cuda::BatchEngine> engine;
        std::unique_ptr<GpuExecutor> executor;
        std::uint32_t region_start{0};
        std::uint32_t region_end{0};
        std::uint32_t next_nonce{0};
        std::uint64_t hashes_done{0};
        std::uint64_t hashes_at_last_report{0};
    };
    std::vector<GpuWorker> gpu_workers_;
    JobLoadedFlag gpu_job_loaded_{};
    bool gpu_pipeline_ready_{false};
#endif
};

} // namespace yerbas::stratum
