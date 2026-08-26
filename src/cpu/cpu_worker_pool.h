#pragma once

#include <array>
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

class WorkerPool {
public:
    // lane_width=0 inherits the process-level grouping width selected by the
    // production tuner. Explicit 1/2/4 values are used by benchmarks/tests.
    // Individual CryptoNight variants still use runtime_cn_widths(), allowing
    // 1/2/4-way execution to vary by variant inside the same hash batch.
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
