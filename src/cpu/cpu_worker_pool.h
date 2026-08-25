#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace yerbas::cpu {

struct Candidate {
    std::uint32_t nonce{0};
};

void set_runtime_lane_width(unsigned int lane_width) noexcept;
unsigned int runtime_lane_width() noexcept;

class WorkerPool {
public:
    // lane_width=0 inherits the process-level runtime policy selected by the
    // production tuner. Explicit 1/2/4 values are used by benchmarks/tests.
    explicit WorkerPool(unsigned int thread_count, unsigned int lane_width = 0);
    ~WorkerPool();

    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;

    unsigned int thread_count() const noexcept;
    unsigned int lane_width() const noexcept;

    std::vector<Candidate> run(const std::array<std::uint8_t, 80>& base_header,
                               const std::array<std::uint8_t, 32>& target_le,
                               std::uint32_t batch_start,
                               unsigned int per_thread);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace yerbas::cpu
