#include "cpu/cpu_worker_pool.h"
#include "cpu/cn_width_tune.h"

#include "ghostrider/ghostrider.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <utility>

namespace yerbas::cpu {
namespace {

void write_nonce(std::array<std::uint8_t, 80>& header, std::uint32_t nonce)
{
    header[76] = static_cast<std::uint8_t>(nonce);
    header[77] = static_cast<std::uint8_t>(nonce >> 8);
    header[78] = static_cast<std::uint8_t>(nonce >> 16);
    header[79] = static_cast<std::uint8_t>(nonce >> 24);
}

bool hash_meets_target(const ghostrider::Hash256& hash,
                       const std::array<std::uint8_t, 32>& target_le)
{
    for (int i = 31; i >= 0; --i) {
        const auto index = static_cast<std::size_t>(i);
        if (hash[index] < target_le[index]) return true;
        if (hash[index] > target_le[index]) return false;
    }
    return true;
}

std::string cn_summary(const ghostrider::StageSchedule& schedule)
{
    std::ostringstream out;
    bool first = true;
    for (const std::uint8_t stage : schedule) {
        if ((stage & ghostrider::kCryptoNightStageFlag) == 0) continue;
        const auto variant = static_cast<std::uint8_t>(stage & 0x7fU);
        if (!first) out << '/';
        out << ghostrider::cryptonight_name(variant);
        first = false;
    }
    return out.str();
}

} // namespace

struct WorkerPool::Impl {
    explicit Impl(unsigned int requested_threads)
        : threads(std::max(1u, requested_threads)), results(threads)
    {
        workers.reserve(threads);
        for (unsigned int index = 0; index < threads; ++index)
            workers.emplace_back([this, index] { worker_loop(index); });
    }

    ~Impl()
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
            ++generation;
        }
        work_ready.notify_all();
        for (auto& worker : workers)
            if (worker.joinable()) worker.join();
    }

    void worker_loop(unsigned int index)
    {
        std::uint64_t seen_generation = 0;
        bool multilane_validated = false;
        bool multilane_disabled = false;

        for (;;) {
            std::array<std::uint8_t, 80> header{};
            std::array<std::uint8_t, 32> target{};
            std::uint32_t start = 0;
            unsigned int count = 0;

            {
                std::unique_lock<std::mutex> lock(mutex);
                work_ready.wait(lock, [&] { return stopping || generation != seen_generation; });
                if (stopping) return;
                seen_generation = generation;
                header = base_header;
                target = target_le;
                start = batch_start + index * per_thread;
                count = per_thread;
            }

            const auto widths = active_cn_widths();
            std::vector<Candidate> found;
            found.reserve(2);

            for (unsigned int i = 0; i < count;) {
                const std::size_t lanes = std::min<std::size_t>(4U, count - i);
                std::array<std::array<std::uint8_t, 80>, 4> lane_headers{};
                std::array<ghostrider::Work, 4> works{};
                std::array<ghostrider::Hash256, 4> hashes{};
                std::array<std::uint32_t, 4> nonces{};

                for (std::size_t lane = 0; lane < lanes; ++lane) {
                    lane_headers[lane] = header;
                    nonces[lane] = start + i + static_cast<unsigned int>(lane);
                    write_nonce(lane_headers[lane], nonces[lane]);
                    works[lane] = {lane_headers[lane].data(), lane_headers[lane].size()};
                }

                bool batch_ok = !multilane_disabled &&
                    ghostrider::hash_optimized_batch(works.data(), hashes.data(), lanes, widths);

                if (batch_ok && !multilane_validated) {
                    bool parity = true;
                    for (std::size_t lane = 0; lane < lanes; ++lane) {
                        const auto reference = ghostrider::hash_optimized(works[lane]);
                        if (reference != hashes[lane]) {
                            parity = false;
                            break;
                        }
                    }
                    if (parity) {
                        multilane_validated = true;
                        if (index == 0)
                            std::cout << "CPU GhostRider multilaned pipeline: parity PASS | production enabled\n";
                    } else {
                        multilane_disabled = true;
                        batch_ok = false;
                        if (index == 0)
                            std::cout << "CPU GhostRider multilaned pipeline: parity FAIL | 1-way fallback\n";
                    }
                }

                if (!batch_ok) {
                    for (std::size_t lane = 0; lane < lanes; ++lane)
                        hashes[lane] = ghostrider::hash_optimized(works[lane]);
                }

                for (std::size_t lane = 0; lane < lanes; ++lane) {
                    if (hash_meets_target(hashes[lane], target))
                        found.push_back({nonces[lane]});
                }
                i += static_cast<unsigned int>(lanes);
            }

            {
                std::lock_guard<std::mutex> lock(mutex);
                results[index] = std::move(found);
                ++completed;
                if (completed == threads) all_done.notify_one();
            }
        }
    }

    std::vector<Candidate> run(const std::array<std::uint8_t, 80>& header,
                               const std::array<std::uint8_t, 32>& target,
                               std::uint32_t start,
                               unsigned int count)
    {
        const auto perf_start = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lock(mutex);
            base_header = header;

            std::array<std::uint8_t, 76> header_key{};
            std::copy_n(header.begin(), header_key.size(), header_key.begin());
            if (!active_target_valid || header_key != active_header_key) {
                active_header_key = header_key;
                active_target = target;
                active_target_valid = true;

                const ghostrider::Work work{header.data(), header.size()};
                active_schedule = ghostrider::stage_schedule_quiet(work);
                active_fingerprint = ghostrider::schedule_fingerprint(active_schedule);
                active_cn_summary = cn_summary(active_schedule);
            }
            target_le = active_target;

            batch_start = start;
            per_thread = count;
            completed = 0;
            for (auto& result : results) result.clear();
            ++generation;
        }
        work_ready.notify_all();

        std::unique_lock<std::mutex> lock(mutex);
        all_done.wait(lock, [&] { return completed == threads; });
        lock.unlock();

        const auto perf_end = std::chrono::steady_clock::now();
        ++run_count;
        if ((run_count % 4U) == 0U) {
            const double elapsed_ms = std::chrono::duration<double, std::milli>(perf_end - perf_start).count();
            const std::uint64_t hashes = static_cast<std::uint64_t>(count) * static_cast<std::uint64_t>(threads);
            const double hps = elapsed_ms > 0.0 ? static_cast<double>(hashes) * 1000.0 / elapsed_ms : 0.0;
            std::cout << std::fixed << std::setprecision(3)
                      << "[CPU GR perf] fingerprint=" << std::hex << std::setw(16) << std::setfill('0')
                      << active_fingerprint << std::dec << std::setfill(' ')
                      << " | CN=" << active_cn_summary
                      << " | threads=" << threads
                      << " | hashes=" << hashes
                      << " | elapsed=" << elapsed_ms << " ms"
                      << " | throughput=" << hps << " H/s"
                      << std::defaultfloat << '\n';
        }

        std::size_t total_candidates = 0;
        for (const auto& result : results) total_candidates += result.size();
        std::vector<Candidate> combined;
        combined.reserve(total_candidates);
        for (auto& result : results)
            combined.insert(combined.end(), result.begin(), result.end());
        return combined;
    }

    const unsigned int threads;
    std::vector<std::thread> workers;
    std::vector<std::vector<Candidate>> results;

    std::mutex mutex;
    std::condition_variable work_ready;
    std::condition_variable all_done;
    bool stopping{false};
    std::uint64_t generation{0};
    unsigned int completed{0};
    std::uint64_t run_count{0};

    std::array<std::uint8_t, 80> base_header{};
    std::array<std::uint8_t, 32> target_le{};
    std::uint32_t batch_start{0};
    unsigned int per_thread{0};

    std::array<std::uint8_t, 76> active_header_key{};
    std::array<std::uint8_t, 32> active_target{};
    bool active_target_valid{false};
    ghostrider::StageSchedule active_schedule{};
    std::uint64_t active_fingerprint{0};
    std::string active_cn_summary;
};

WorkerPool::WorkerPool(unsigned int thread_count)
    : impl_(std::make_unique<Impl>(thread_count))
{
}

WorkerPool::~WorkerPool() = default;

unsigned int WorkerPool::thread_count() const noexcept
{
    return impl_->threads;
}

std::vector<Candidate> WorkerPool::run(const std::array<std::uint8_t, 80>& base_header,
                                       const std::array<std::uint8_t, 32>& target_le,
                                       std::uint32_t batch_start,
                                       unsigned int per_thread)
{
    return impl_->run(base_header, target_le, batch_start, per_thread);
}

} // namespace yerbas::cpu
