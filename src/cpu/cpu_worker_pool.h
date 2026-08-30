#pragma once

#include "cpu/cpu_topology.h"
#include "miner.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace yerbas::cpu {

struct Candidate {
    std::uint32_t nonce{0};
};

using CnWidthPolicy = std::array<unsigned int, 6>;

void set_runtime_lane_width(unsigned int lane_width) noexcept;
unsigned int runtime_lane_width() noexcept;
void set_runtime_cn_widths(const CnWidthPolicy& widths) noexcept;
CnWidthPolicy runtime_cn_widths() noexcept;

// Autotune measurements use the same production worker pool, but they must not
// update live fingerprint-learning or stage-profile caches.
void set_tuning_measurement_mode(bool enabled) noexcept;
bool tuning_measurement_mode() noexcept;

class WorkerPool {
public:
    // lane_width=0 inherits the process-level grouping width selected by the
    // production tuner. The affinity policy is captured when the pool is
    // constructed so later tuner/global state changes cannot alter or misreport
    // where these persistent worker threads are actually running.
    explicit WorkerPool(unsigned int thread_count,
                        unsigned int lane_width = 0,
                        AffinityPolicy affinity = runtime_affinity_policy());
    ~WorkerPool();

    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    unsigned int thread_count() const noexcept;
    unsigned int lane_width() const noexcept;
    AffinityPolicy affinity_policy() const noexcept;

    // Production calls inherit the process-wide Ctrl+C flag. Autotune callers
    // may pass their own stop flag explicitly; nullptr disables interruption.
    std::vector<Candidate> run(const std::array<std::uint8_t, 80>& base_header,
                               const std::array<std::uint8_t, 32>& target_le,
                               std::uint32_t batch_start,
                               unsigned int per_thread,
                               const std::atomic_bool* stop = &yerbas::g_stop_requested);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace yerbas::cpu
