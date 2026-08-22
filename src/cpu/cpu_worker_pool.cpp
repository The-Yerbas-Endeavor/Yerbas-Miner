#include "cpu/cpu_worker_pool.h"

#include "ghostrider/ghostrider.h"
#ifdef YERBAS_HAS_CUDA
#include "cuda/cuda_backend.h"
#endif

#include <algorithm>
#include <condition_variable>
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

std::string display_hex(const std::array<std::uint8_t, 32>& value_le)
{
    std::ostringstream ss;
    ss << std::hex << std::setfill('0');
    for (auto it = value_le.rbegin(); it != value_le.rend(); ++it)
        ss << std::setw(2) << static_cast<unsigned int>(*it);
    return ss.str();
}

std::string display_hex64(const std::array<std::uint8_t, 64>& value)
{
    std::ostringstream ss;
    ss << std::hex << std::setfill('0');
    for (const auto byte : value) ss << std::setw(2) << static_cast<unsigned int>(byte);
    return ss.str();
}

std::string nonce_hex(std::uint32_t nonce)
{
    std::ostringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(8) << nonce;
    return ss.str();
}

const char* stage_name(std::uint8_t stage)
{
    static constexpr const char* core_names[15] = {
        "BLAKE-512", "BMW-512", "Groestl-512", "JH-512", "Keccak-512",
        "Skein-512", "Luffa-512", "CubeHash-512", "Shavite-512", "SIMD-512",
        "Echo-512", "Hamsi-512", "Fugue-512", "Shabal-512", "Whirlpool"
    };
    static constexpr const char* cn_names[6] = {
        "CN-Dark", "CN-DarkLite", "CN-Fast", "CN-Lite", "CN-Turtle", "CN-TurtleLite"
    };
    if ((stage & ghostrider::kCryptoNightStageFlag) != 0) {
        const auto index = static_cast<std::uint8_t>(stage & 0x7fU);
        return index < 6 ? cn_names[index] : "CN?";
    }
    return stage < 15 ? core_names[stage] : "Core?";
}

#ifdef YERBAS_HAS_CUDA
std::array<std::uint8_t, 64> cuda_core_stage(int device_id,
                                             const std::uint8_t* input,
                                             std::size_t length,
                                             std::uint8_t algorithm)
{
    switch (algorithm) {
    case 0: return cuda::blake512_reference_stage(device_id, input, length);
    case 1: return cuda::bmw512_reference_stage(device_id, input, length);
    case 2: return cuda::groestl512_reference_stage(device_id, input, length);
    case 3: return cuda::jh512_reference_stage(device_id, input, length);
    case 4: return cuda::keccak512_reference_stage(device_id, input, length);
    case 5: return cuda::skein512_reference_stage(device_id, input, length);
    case 6: return cuda::luffa512_reference_stage(device_id, input, length);
    case 7: return cuda::cubehash512_reference_stage(device_id, input, length);
    case 8: return cuda::shavite512_reference_stage(device_id, input, length);
    case 9: return cuda::simd512_reference_stage(device_id, input, length);
    case 10: return cuda::echo512_reference_stage(device_id, input, length);
    case 11: return cuda::hamsi512_reference_stage(device_id, input, length);
    case 12: return cuda::fugue512_reference_stage(device_id, input, length);
    case 13: return cuda::shabal512_reference_stage(device_id, input, length);
    case 14: return cuda::whirlpool512_reference_stage(device_id, input, length);
    default: throw std::runtime_error("invalid CUDA core stage index");
    }
}

void trace_first_divergence_prefix(const std::array<std::uint8_t, 80>& header,
                                   std::uint32_t nonce,
                                   const ghostrider::StageSchedule& schedule)
{
    auto traced_header = header;
    write_nonce(traced_header, nonce);

    ghostrider::Hash512 cpu_state{};
    ghostrider::Hash512 gpu_state{};

    std::cout << "[CPU/CUDA TRACE] nonce=" << nonce_hex(nonce)
              << " | begin stage-by-stage prefix comparison\n";

    for (std::size_t i = 0; i < schedule.size(); ++i) {
        const std::uint8_t stage = schedule[i];
        if ((stage & ghostrider::kCryptoNightStageFlag) != 0) {
            std::cout << "[CPU/CUDA TRACE] stage " << i << ' ' << stage_name(stage)
                      << " | first unresolved CryptoNight boundary after matched core prefix\n";
            break;
        }

        if (i == 0) {
            const ghostrider::Work cpu_work{traced_header.data(), traced_header.size()};
            cpu_state = ghostrider::stage_reference(cpu_work, stage);
            gpu_state = cuda_core_stage(0, traced_header.data(), traced_header.size(), stage);
        } else {
            const ghostrider::Work cpu_work{cpu_state.data(), cpu_state.size()};
            cpu_state = ghostrider::stage_reference(cpu_work, stage);
            gpu_state = cuda_core_stage(0, gpu_state.data(), gpu_state.size(), stage);
        }

        const bool match = cpu_state == gpu_state;
        std::cout << "[CPU/CUDA TRACE] stage " << i << ' ' << stage_name(stage)
                  << " | " << (match ? "MATCH" : "MISMATCH") << '\n';
        if (!match) {
            std::cout << "[CPU/CUDA TRACE] CPU=" << display_hex64(cpu_state) << '\n'
                      << "[CPU/CUDA TRACE] GPU=" << display_hex64(gpu_state) << '\n';
            break;
        }
    }
}
#endif

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
        for (auto& result : results) combined.insert(combined.end(), result.begin(), result.end());
        lock.unlock();

        std::vector<Candidate> verified;
        verified.reserve(combined.size());
        for (const auto& candidate : combined) {
            auto verify_header = header;
            write_nonce(verify_header, candidate.nonce);
            const ghostrider::Work work{verify_header.data(), verify_header.size()};
            const auto cpu_hash = ghostrider::hash_reference(work);
            const auto staged_hash = ghostrider::hash_staged_reference(work);
            const bool local_pass = hash_meets_target(cpu_hash, target);

#ifdef YERBAS_HAS_CUDA
            bool parity_ready = false;
            bool parity_match = false;
            bool staged_match = false;
            bool cuda_pass = false;
            std::array<std::uint8_t, 32> cuda_hash{};
            ghostrider::StageSchedule schedule{};
            if (local_pass && cuda::device_count() > 0) {
                try {
                    if (!parity_engine) parity_engine = std::make_unique<cuda::BatchEngine>(0, 1);
                    cuda::JobDescriptor parity_job{};
                    parity_job.header = header;
                    parity_job.target_le.fill(0xff);
                    schedule = ghostrider::stage_schedule(work);
                    parity_job.stages = schedule;
                    parity_engine->upload_job(parity_job);
                    const auto parity_candidates = parity_engine->scan(candidate.nonce);
                    const auto hit = std::find_if(parity_candidates.begin(), parity_candidates.end(),
                                                  [&](const cuda::Candidate& c) { return c.nonce == candidate.nonce; });
                    if (hit != parity_candidates.end()) {
                        cuda_hash = hit->hash;
                        parity_ready = true;
                        parity_match = (cuda_hash == cpu_hash);
                        staged_match = (cuda_hash == staged_hash);
                        cuda_pass = hash_meets_target(cuda_hash, target);
                    }
                } catch (const std::exception& e) {
                    std::cout << "[CPU/CUDA PARITY] nonce=" << nonce_hex(candidate.nonce)
                              << " | ERROR=" << e.what() << '\n';
                }
            }

            std::cout << "[CPU verify] nonce=" << nonce_hex(candidate.nonce)
                      << " | canonical=" << display_hex(cpu_hash)
                      << " | staged=" << display_hex(staged_hash)
                      << " | target=" << display_hex(target)
                      << " | local=" << (local_pass ? "PASS" : "FAIL");
            if (parity_ready) {
                std::cout << " | CUDA=" << display_hex(cuda_hash)
                          << " | canonical/CUDA=" << (parity_match ? "MATCH" : "MISMATCH")
                          << " | staged/CUDA=" << (staged_match ? "MATCH" : "MISMATCH")
                          << " | cuda_target=" << (cuda_pass ? "PASS" : "FAIL");
            } else if (local_pass) {
                std::cout << " | parity=UNAVAILABLE";
            }

            const bool submit_ok = local_pass && (!parity_ready || (parity_match && cuda_pass));
            std::cout << (submit_ok ? " | pool=SUBMIT" : " | pool=HOLD") << '\n';

            if (parity_ready && !parity_match && !divergence_trace_emitted) {
                divergence_trace_emitted = true;
                trace_first_divergence_prefix(header, candidate.nonce, schedule);
            }

            if (submit_ok) verified.push_back(candidate);
#else
            std::cout << "[CPU verify] nonce=" << nonce_hex(candidate.nonce)
                      << " | canonical=" << display_hex(cpu_hash)
                      << " | staged=" << display_hex(staged_hash)
                      << " | target=" << display_hex(target)
                      << " | local=" << (local_pass ? "PASS" : "FAIL")
                      << (local_pass ? " | pool=SUBMIT" : "") << '\n';
            if (local_pass) verified.push_back(candidate);
#endif
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
#ifdef YERBAS_HAS_CUDA
    std::unique_ptr<cuda::BatchEngine> parity_engine;
    bool divergence_trace_emitted{false};
#endif
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
