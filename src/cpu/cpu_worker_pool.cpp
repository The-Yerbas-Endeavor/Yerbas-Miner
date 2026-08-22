#include "cpu/cpu_worker_pool.h"

#include "ghostrider/ghostrider.h"

#include <algorithm>
#include <condition_variable>
#include <iomanip>
#include <iostream>
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

std::string display_hex(const std::array<std::uint8_t, 32>& value_le)
{
    std::ostringstream ss;
    ss << std::hex << std::setfill('0');
    for (auto it = value_le.rbegin(); it != value_le.rend(); ++it)
        ss << std::setw(2) << static_cast<unsigned int>(*it);
    return ss.str();
}

std::string nonce_hex(std::uint32_t nonce)
{
    std::ostringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(8) << nonce;
    return ss.str();
}

} // namespace

struct WorkerPool::Impl {
    explicit Impl(unsigned int requested_threads)
        : threads(std::max(1u, requested_threads)), results(threads)
    {
        workers.reserve(threads);
        for (unsigned int index = 0; index < threads; ++index) {
            workers.emplace_back([this, index] { worker_loop(index); });
        }
    }

    ~Impl()
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
            ++generation;
        }
        work_ready.notify_all();
        for (auto& worker : workers) {
            if (worker.joinable()) worker.join();
        }
    }

    void worker_loop(unsigned int index)
    {
        std::uint64_t seen_generation = 0;
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

            std::vector<Candidate> found;
            found.reserve(2);
            for (unsigned int i = 0; i < count; ++i) {
                const std::uint32_t nonce = start + i;
                write_nonce(header, nonce);
                const ghostrider::Work work{header.data(), header.size()};
                const auto hash = ghostrider::hash_reference(work);
                if (hash_meets_target(hash, target)) found.push_back({nonce});
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
        {
            std::lock_guard<std::mutex> lock(mutex);
            base_header = header;
            target_le = target;
            batch_start = start;
            per_thread = count;
            completed = 0;
            for (auto& result : results) result.clear();
            ++generation;
        }
        work_ready.notify_all();

        std::unique_lock<std::mutex> lock(mutex);
        all_done.wait(lock, [&] { return completed == threads; });

        std::size_t total_candidates = 0;
        for (const auto& result : results) total_candidates += result.size();
        std::vector<Candidate> combined;
        combined.reserve(total_candidates);
        for (auto& result : results) {
            combined.insert(combined.end(), result.begin(), result.end());
        }
        lock.unlock();

        // Re-hash every candidate after all worker threads are idle, then log
        // the exact final 256-bit hash and target in conventional big-endian
        // display order. During hybrid mining the CPU owns the upper half of
        // the nonce space (bit 31 set). The live pool test showed every such
        // CPU candidate being rejected while both lower-half CUDA regions were
        // accepted. Until CPU/pool parity for high-bit nonces is proven, do not
        // send those known-bad candidates to the pool. CPU-only mining starts
        // below 0x80000000 and therefore remains available for the parity test.
        std::vector<Candidate> verified;
        verified.reserve(combined.size());
        for (const auto& candidate : combined) {
            auto verify_header = header;
            write_nonce(verify_header, candidate.nonce);
            const ghostrider::Work work{verify_header.data(), verify_header.size()};
            const auto hash = ghostrider::hash_reference(work);
            const bool local_pass = hash_meets_target(hash, target);
            const bool high_bit_nonce = (candidate.nonce & 0x80000000U) != 0U;
            std::cout << "[CPU verify] nonce=" << nonce_hex(candidate.nonce)
                      << " | hash=" << display_hex(hash)
                      << " | target=" << display_hex(target)
                      << " | local=" << (local_pass ? "PASS" : "FAIL");
            if (local_pass && high_bit_nonce) {
                std::cout << " | pool=QUARANTINE-high-bit";
            }
            std::cout << '\n';
            if (local_pass && !high_bit_nonce) verified.push_back(candidate);
        }
        return verified;
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

    std::array<std::uint8_t, 80> base_header{};
    std::array<std::uint8_t, 32> target_le{};
    std::uint32_t batch_start{0};
    unsigned int per_thread{0};
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
