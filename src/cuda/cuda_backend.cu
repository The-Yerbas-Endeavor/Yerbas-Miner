#include "cuda/cuda_backend.h"
#include "cuda/core/stage_dispatch.cuh"
#include "cuda/core_coverage.h"
#include "cuda/cryptonight/cn_config.cuh"
#include "cuda/cryptonight/cn_final.cuh"
#include "cuda/cryptonight/cn_slow_hash.cuh"
#include "cuda/cryptonight/cn_split.cuh"
#include "ghostrider/ghostrider.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace yerbas::cuda {
namespace {

constexpr std::size_t kMaxCandidates = 64;
constexpr std::size_t kStateBytes = 64;
constexpr std::size_t kCnScratchpadStride = cryptonight::max_scratchpad_bytes();

struct DeviceJob {
    std::uint8_t header[80];
    std::uint8_t target_le[32];
    std::uint8_t stages[18];
};

struct DeviceCandidate {
    std::uint32_t nonce;
    std::uint8_t hash[32];
};

void check_cuda(cudaError_t status, const char* what)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

__device__ bool hash_meets_target(const std::uint8_t* hash,
                                  const std::uint8_t* target_le)
{
    for (int i = 31; i >= 0; --i) {
        if (hash[i] < target_le[i]) return true;
        if (hash[i] > target_le[i]) return false;
    }
    return true;
}

__global__ void initialize_nonce_batch(std::uint32_t start_nonce,
                                       std::uint32_t* nonces,
                                       std::size_t count)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        nonces[index] = start_nonce + static_cast<std::uint32_t>(index);
    }
}

__global__ void core512_stage(const DeviceJob* job,
                              const std::uint32_t* nonces,
                              std::uint8_t* states,
                              std::size_t count,
                              int stage_index,
                              std::uint8_t algorithm)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;

    std::uint8_t digest[64];
    bool ok = false;
    if (stage_index == 0) {
        std::uint8_t header[80];
        #pragma unroll
        for (int i = 0; i < 80; ++i) header[i] = job->header[i];
        const std::uint32_t nonce = nonces[index];
        header[76] = static_cast<std::uint8_t>(nonce);
        header[77] = static_cast<std::uint8_t>(nonce >> 8);
        header[78] = static_cast<std::uint8_t>(nonce >> 16);
        header[79] = static_cast<std::uint8_t>(nonce >> 24);
        ok = core::dispatch_core512(algorithm, header, 80, digest);
    } else {
        ok = core::dispatch_core512(algorithm,
                                    states + index * kStateBytes,
                                    kStateBytes,
                                    digest);
    }

    if (!ok) return;
    std::uint8_t* output = states + index * kStateBytes;
    #pragma unroll
    for (int i = 0; i < 64; ++i) output[i] = digest[i];
}

__global__ void cryptonight_stage(std::uint8_t* states,
                                  std::size_t count,
                                  std::uint8_t variant,
                                  std::uint8_t* scratchpads)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;

    const auto cfg = cryptonight::config_value(variant);
    const std::size_t scratchpad_stride = static_cast<std::size_t>(cfg.page_size);
    std::uint8_t* state = states + index * kStateBytes;
    std::uint8_t* scratchpad = scratchpads + index * scratchpad_stride;
    std::uint8_t digest[32];

    if (!cryptonight::slow_hash(variant, state, kStateBytes, scratchpad, digest)) return;

    #pragma unroll
    for (int i = 0; i < 32; ++i) state[i] = digest[i];
    #pragma unroll
    for (int i = 32; i < 64; ++i) state[i] = 0;
}

template <std::uint8_t VariantIndex>
__global__ void cryptonight_setup_stage(std::uint8_t* states,
                                        std::size_t count,
                                        std::uint8_t* scratchpads,
                                        cryptonight::SplitContext* contexts)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;

    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    std::uint8_t* state = states + index * kStateBytes;
    std::uint8_t* scratchpad = scratchpads + index * static_cast<std::size_t>(cfg.page_size);
    cryptonight::split_setup<VariantIndex>(state, kStateBytes, scratchpad, contexts[index]);
}

template <std::uint8_t VariantIndex>
__global__ void cryptonight_loop_stage(std::size_t count,
                                       std::uint8_t* scratchpads,
                                       const cryptonight::SplitContext* contexts)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;

    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    std::uint8_t* scratchpad = scratchpads + index * static_cast<std::size_t>(cfg.page_size);
    cryptonight::split_memory_loop<VariantIndex>(scratchpad, contexts[index]);
}

template <std::uint8_t VariantIndex>
__global__ void cryptonight_final_stage(std::uint8_t* states,
                                        std::size_t count,
                                        std::uint8_t* scratchpads,
                                        cryptonight::SplitContext* contexts)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;

    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    std::uint8_t* state = states + index * kStateBytes;
    std::uint8_t* scratchpad = scratchpads + index * static_cast<std::size_t>(cfg.page_size);
    std::uint8_t digest[32];
    cryptonight::split_finalize<VariantIndex>(scratchpad, contexts[index], digest);

    #pragma unroll
    for (int i = 0; i < 32; ++i) state[i] = digest[i];
    #pragma unroll
    for (int i = 32; i < 64; ++i) state[i] = 0;
}

template <std::uint8_t VariantIndex>
void launch_split_cryptonight_variant(cudaStream_t stream,
                                      std::uint8_t* states,
                                      std::size_t count,
                                      std::uint8_t* scratchpads,
                                      cryptonight::SplitContext* contexts)
{
    // Keep the production path on the full-pipeline-proven geometry.
    // The CN-Fast microbenchmark favored 64/128/128, but applying that
    // geometry to every GhostRider CN variant regressed the real pipeline.
    constexpr int setup_threads = 64;
    constexpr int loop_threads = 64;
    constexpr int final_threads = 64;
    const int setup_blocks = static_cast<int>((count + setup_threads - 1) / setup_threads);
    const int loop_blocks = static_cast<int>((count + loop_threads - 1) / loop_threads);
    const int final_blocks = static_cast<int>((count + final_threads - 1) / final_threads);

    cryptonight_setup_stage<VariantIndex><<<setup_blocks, setup_threads, 0, stream>>>(states, count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "GhostRider CUDA CryptoNight setup launch failed");
    cryptonight_loop_stage<VariantIndex><<<loop_blocks, loop_threads, 0, stream>>>(count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "GhostRider CUDA CryptoNight loop launch failed");
    cryptonight_final_stage<VariantIndex><<<final_blocks, final_threads, 0, stream>>>(states, count, scratchpads, contexts);
    check_cuda(cudaGetLastError(), "GhostRider CUDA CryptoNight final launch failed");
}

void launch_split_cryptonight(cudaStream_t stream,
                              std::uint8_t* states,
                              std::size_t count,
                              std::uint8_t variant,
                              std::uint8_t* scratchpads,
                              cryptonight::SplitContext* contexts)
{
    switch (variant) {
    case 0: launch_split_cryptonight_variant<0>(stream, states, count, scratchpads, contexts); break;
    case 1: launch_split_cryptonight_variant<1>(stream, states, count, scratchpads, contexts); break;
    case 2: launch_split_cryptonight_variant<2>(stream, states, count, scratchpads, contexts); break;
    case 3: launch_split_cryptonight_variant<3>(stream, states, count, scratchpads, contexts); break;
    case 4: launch_split_cryptonight_variant<4>(stream, states, count, scratchpads, contexts); break;
    case 5: launch_split_cryptonight_variant<5>(stream, states, count, scratchpads, contexts); break;
    default: throw std::runtime_error("GhostRider CryptoNight stage has invalid variant index");
    }
}

__global__ void keccak512_validation_kernel(const std::uint8_t* input,
                                            std::size_t length,
                                            std::uint8_t* output)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    std::uint8_t digest[64];
    core::keccak512(input, length, digest);
    #pragma unroll
    for (int i = 0; i < 64; ++i) output[i] = digest[i];
}

__global__ void cubehash512_validation_kernel(const std::uint8_t* input,
                                              std::size_t length,
                                              std::uint8_t* output)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    std::uint8_t digest[64];
    core::cubehash512(input, length, digest);
    #pragma unroll
    for (int i = 0; i < 64; ++i) output[i] = digest[i];
}

__global__ void collect_candidates(const std::uint32_t* nonces,
                                   const std::uint8_t* states,
                                   std::size_t count,
                                   const DeviceJob* job,
                                   DeviceCandidate* candidates,
                                   unsigned int* candidate_count)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;

    const std::uint8_t* hash = states + index * kStateBytes;
    if (!hash_meets_target(hash, job->target_le)) return;

    const unsigned int slot = atomicAdd(candidate_count, 1U);
    if (slot >= kMaxCandidates) return;

    candidates[slot].nonce = nonces[index];
    for (int i = 0; i < 32; ++i) candidates[slot].hash[i] = hash[i];
}

Hash512 run_validation_kernel(int device_id,
                              const std::uint8_t* input,
                              std::size_t length,
                              bool use_keccak)
{
    if (input == nullptr || length == 0) {
        throw std::invalid_argument("CUDA core validation input must not be empty");
    }

    check_cuda(cudaSetDevice(device_id), "cudaSetDevice failed");
    std::uint8_t* d_input = nullptr;
    std::uint8_t* d_output = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_input), length),
               "cudaMalloc validation input failed");
    try {
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_output), 64),
                   "cudaMalloc validation output failed");
        check_cuda(cudaMemcpy(d_input, input, length, cudaMemcpyHostToDevice),
                   "cudaMemcpy validation input failed");

        if (use_keccak) {
            keccak512_validation_kernel<<<1, 1>>>(d_input, length, d_output);
        } else {
            cubehash512_validation_kernel<<<1, 1>>>(d_input, length, d_output);
        }
        check_cuda(cudaGetLastError(), "CUDA core validation launch failed");
        check_cuda(cudaDeviceSynchronize(), "CUDA core validation synchronize failed");

        Hash512 out{};
        check_cuda(cudaMemcpy(out.data(), d_output, out.size(), cudaMemcpyDeviceToHost),
                   "cudaMemcpy validation output failed");
        cudaFree(d_output);
        cudaFree(d_input);
        return out;
    } catch (...) {
        if (d_output) cudaFree(d_output);
        if (d_input) cudaFree(d_input);
        throw;
    }
}

} // namespace

struct BatchEngine::Impl {
    int device_id{-1};
    std::size_t batch_size{0};
    unsigned int fallback_threads{1};
    cudaStream_t stream{};
    DeviceJob* d_job{nullptr};
    std::uint32_t* d_nonces{nullptr};
    std::uint8_t* d_states{nullptr};
    std::uint8_t* d_cn_scratchpads{nullptr};
    cryptonight::SplitContext* d_cn_contexts{nullptr};
    DeviceCandidate* d_candidates{nullptr};
    unsigned int* d_candidate_count{nullptr};
    JobDescriptor host_job{};
    std::vector<std::uint8_t> host_states;
    bool job_loaded{false};

    Impl(int id, std::size_t requested_size, unsigned int requested_fallback_threads)
        : device_id(id),
          batch_size(std::min(requested_size, cryptonight::kInitialMaxBatch)),
          fallback_threads(std::max(1u, requested_fallback_threads)),
          host_states(batch_size * kStateBytes)
    {
        if (batch_size == 0) throw std::runtime_error("CUDA batch size must be greater than zero");
        check_cuda(cudaSetDevice(device_id), "cudaSetDevice failed");
        check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_job), sizeof(DeviceJob)),
                   "cudaMalloc job failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_nonces), batch_size * sizeof(std::uint32_t)),
                   "cudaMalloc nonce batch failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_states), batch_size * kStateBytes),
                   "cudaMalloc 512-bit state batch failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_cn_scratchpads),
                              batch_size * kCnScratchpadStride),
                   "cudaMalloc CryptoNight scratchpads failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_cn_contexts),
                              batch_size * sizeof(cryptonight::SplitContext)),
                   "cudaMalloc CryptoNight split contexts failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_candidates), kMaxCandidates * sizeof(DeviceCandidate)),
                   "cudaMalloc candidate buffer failed");
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_candidate_count), sizeof(unsigned int)),
                   "cudaMalloc candidate counter failed");
    }

    ~Impl()
    {
        cudaSetDevice(device_id);
        if (d_candidate_count) cudaFree(d_candidate_count);
        if (d_candidates) cudaFree(d_candidates);
        if (d_cn_contexts) cudaFree(d_cn_contexts);
        if (d_cn_scratchpads) cudaFree(d_cn_scratchpads);
        if (d_states) cudaFree(d_states);
        if (d_nonces) cudaFree(d_nonces);
        if (d_job) cudaFree(d_job);
        if (stream) cudaStreamDestroy(stream);
    }
};

BatchEngine::BatchEngine(int device_id,
                         std::size_t batch_size,
                         unsigned int fallback_threads)
    : impl_(std::make_unique<Impl>(device_id, batch_size, fallback_threads))
{
}
BatchEngine::~BatchEngine() = default;
BatchEngine::BatchEngine(BatchEngine&&) noexcept = default;
BatchEngine& BatchEngine::operator=(BatchEngine&&) noexcept = default;

void BatchEngine::upload_job(const JobDescriptor& job)
{
    impl_->host_job = job;
    DeviceJob device_job{};
    std::memcpy(device_job.header, job.header.data(), job.header.size());
    std::memcpy(device_job.target_le, job.target_le.data(), job.target_le.size());
    std::memcpy(device_job.stages, job.stages.data(), job.stages.size());
    check_cuda(cudaSetDevice(impl_->device_id), "cudaSetDevice failed");
    check_cuda(cudaMemcpyAsync(impl_->d_job, &device_job, sizeof(device_job),
                               cudaMemcpyHostToDevice, impl_->stream),
               "cudaMemcpyAsync job failed");
    check_cuda(cudaStreamSynchronize(impl_->stream), "cudaStreamSynchronize job failed");
    impl_->job_loaded = true;
}

std::vector<Candidate> BatchEngine::scan(std::uint32_t start_nonce)
{
    if (!impl_->job_loaded) throw std::runtime_error("CUDA batch scan requested before upload_job");
    if (!hash_pipeline_ready()) {
        throw std::runtime_error("CUDA GhostRider scan requested before full native CUDA coverage is available");
    }

    check_cuda(cudaSetDevice(impl_->device_id), "cudaSetDevice failed");
    check_cuda(cudaMemsetAsync(impl_->d_candidate_count, 0, sizeof(unsigned int), impl_->stream),
               "cudaMemsetAsync candidate counter failed");

    constexpr int core_threads = 256;
    constexpr int cn_threads = 64;
    const int core_blocks = static_cast<int>((impl_->batch_size + core_threads - 1) / core_threads);
    const int cn_blocks = static_cast<int>((impl_->batch_size + cn_threads - 1) / cn_threads);
    initialize_nonce_batch<<<core_blocks, core_threads, 0, impl_->stream>>>(start_nonce,
                                                                           impl_->d_nonces,
                                                                           impl_->batch_size);
    check_cuda(cudaGetLastError(), "initialize_nonce_batch launch failed");

    for (int stage_index = 0; stage_index < 18; ++stage_index) {
        const std::uint8_t stage = impl_->host_job.stages[static_cast<std::size_t>(stage_index)];
        const bool is_cn = (stage & ghostrider::kCryptoNightStageFlag) != 0;
        const std::uint8_t algorithm = static_cast<std::uint8_t>(stage & 0x7fU);

        if (is_cn) {
            if (algorithm >= cryptonight::kVariantConfigs.size()) {
                throw std::runtime_error("GhostRider CryptoNight stage has invalid variant index");
            }
            launch_split_cryptonight(impl_->stream,
                                     impl_->d_states,
                                     impl_->batch_size,
                                     algorithm,
                                     impl_->d_cn_scratchpads,
                                     impl_->d_cn_contexts);
        } else {
            if (!core::core512_implemented(algorithm)) {
                throw std::runtime_error("GhostRider conventional stage has no native CUDA implementation");
            }
            core512_stage<<<core_blocks, core_threads, 0, impl_->stream>>>(impl_->d_job,
                                                                           impl_->d_nonces,
                                                                           impl_->d_states,
                                                                           impl_->batch_size,
                                                                           stage_index,
                                                                           algorithm);
            check_cuda(cudaGetLastError(), "GhostRider CUDA core stage launch failed");
        }
    }

    collect_candidates<<<core_blocks, core_threads, 0, impl_->stream>>>(impl_->d_nonces,
                                                                        impl_->d_states,
                                                                        impl_->batch_size,
                                                                        impl_->d_job,
                                                                        impl_->d_candidates,
                                                                        impl_->d_candidate_count);
    check_cuda(cudaGetLastError(), "collect_candidates launch failed");

    unsigned int count = 0;
    check_cuda(cudaMemcpyAsync(&count, impl_->d_candidate_count, sizeof(count),
                               cudaMemcpyDeviceToHost, impl_->stream),
               "cudaMemcpyAsync candidate count failed");
    check_cuda(cudaStreamSynchronize(impl_->stream), "cudaStreamSynchronize scan failed");

    count = std::min<unsigned int>(count, static_cast<unsigned int>(kMaxCandidates));
    if (count == 0) return {};

    std::vector<DeviceCandidate> raw(count);
    check_cuda(cudaMemcpy(raw.data(), impl_->d_candidates,
                          count * sizeof(DeviceCandidate), cudaMemcpyDeviceToHost),
               "cudaMemcpy candidates failed");

    std::vector<Candidate> out;
    out.reserve(count);
    for (const auto& item : raw) {
        Candidate candidate;
        candidate.nonce = item.nonce;
        std::copy(std::begin(item.hash), std::end(item.hash), candidate.hash.begin());
        out.push_back(candidate);
    }
    return out;
}

std::vector<Candidate> BatchEngine::scan_profiled(std::uint32_t start_nonce, BatchProfile& profile)
{
    if (!impl_->job_loaded) throw std::runtime_error("CUDA profiled batch scan requested before upload_job");
    if (!hash_pipeline_ready()) {
        throw std::runtime_error("CUDA GhostRider scan requested before full native CUDA coverage is available");
    }

    profile = BatchProfile{};
    profile.hashes = impl_->batch_size;

    check_cuda(cudaSetDevice(impl_->device_id), "cudaSetDevice failed");
    check_cuda(cudaMemsetAsync(impl_->d_candidate_count, 0, sizeof(unsigned int), impl_->stream),
               "cudaMemsetAsync candidate counter failed");

    constexpr int core_threads = 256;
    constexpr int cn_threads = 64;
    const int core_blocks = static_cast<int>((impl_->batch_size + core_threads - 1) / core_threads);
    const int cn_blocks = static_cast<int>((impl_->batch_size + cn_threads - 1) / cn_threads);

    cudaEvent_t total_start{}, total_stop{}, section_start{}, section_stop{};
    check_cuda(cudaEventCreate(&total_start), "cudaEventCreate total_start failed");
    check_cuda(cudaEventCreate(&total_stop), "cudaEventCreate total_stop failed");
    check_cuda(cudaEventCreate(&section_start), "cudaEventCreate section_start failed");
    check_cuda(cudaEventCreate(&section_stop), "cudaEventCreate section_stop failed");

    auto elapsed = [&](float& out) {
        check_cuda(cudaEventRecord(section_stop, impl_->stream), "cudaEventRecord section stop failed");
        check_cuda(cudaEventSynchronize(section_stop), "cudaEventSynchronize section stop failed");
        check_cuda(cudaEventElapsedTime(&out, section_start, section_stop), "cudaEventElapsedTime failed");
    };

    try {
        check_cuda(cudaEventRecord(total_start, impl_->stream), "cudaEventRecord total start failed");
        check_cuda(cudaEventRecord(section_start, impl_->stream), "cudaEventRecord nonce start failed");
        initialize_nonce_batch<<<core_blocks, core_threads, 0, impl_->stream>>>(start_nonce,
                                                                               impl_->d_nonces,
                                                                               impl_->batch_size);
        check_cuda(cudaGetLastError(), "initialize_nonce_batch launch failed");
        elapsed(profile.nonce_init_ms);

        for (int stage_index = 0; stage_index < 18; ++stage_index) {
            const std::uint8_t stage = impl_->host_job.stages[static_cast<std::size_t>(stage_index)];
            const bool is_cn = (stage & ghostrider::kCryptoNightStageFlag) != 0;
            const std::uint8_t algorithm = static_cast<std::uint8_t>(stage & 0x7fU);

            check_cuda(cudaEventRecord(section_start, impl_->stream), "cudaEventRecord stage start failed");
            if (is_cn) {
                if (algorithm >= cryptonight::kVariantConfigs.size()) {
                    throw std::runtime_error("GhostRider CryptoNight stage has invalid variant index");
                }
                launch_split_cryptonight(impl_->stream,
                                         impl_->d_states,
                                         impl_->batch_size,
                                         algorithm,
                                         impl_->d_cn_scratchpads,
                                         impl_->d_cn_contexts);
            } else {
                if (!core::core512_implemented(algorithm)) {
                    throw std::runtime_error("GhostRider conventional stage has no native CUDA implementation");
                }
                core512_stage<<<core_blocks, core_threads, 0, impl_->stream>>>(impl_->d_job,
                                                                               impl_->d_nonces,
                                                                               impl_->d_states,
                                                                               impl_->batch_size,
                                                                               stage_index,
                                                                               algorithm);
                check_cuda(cudaGetLastError(), "GhostRider CUDA core stage launch failed");
            }

            float stage_ms = 0.0F;
            elapsed(stage_ms);
            profile.stages[static_cast<std::size_t>(stage_index)] = StageTiming{stage_index, stage, stage_ms};
            profile.stage_total_ms += stage_ms;
        }

        check_cuda(cudaEventRecord(section_start, impl_->stream), "cudaEventRecord candidate start failed");
        collect_candidates<<<core_blocks, core_threads, 0, impl_->stream>>>(impl_->d_nonces,
                                                                            impl_->d_states,
                                                                            impl_->batch_size,
                                                                            impl_->d_job,
                                                                            impl_->d_candidates,
                                                                            impl_->d_candidate_count);
        check_cuda(cudaGetLastError(), "collect_candidates launch failed");
        elapsed(profile.candidate_ms);

        check_cuda(cudaEventRecord(total_stop, impl_->stream), "cudaEventRecord total stop failed");
        check_cuda(cudaEventSynchronize(total_stop), "cudaEventSynchronize total stop failed");
        check_cuda(cudaEventElapsedTime(&profile.total_gpu_ms, total_start, total_stop),
                   "cudaEventElapsedTime total failed");

        unsigned int count = 0;
        check_cuda(cudaMemcpyAsync(&count, impl_->d_candidate_count, sizeof(count),
                                   cudaMemcpyDeviceToHost, impl_->stream),
                   "cudaMemcpyAsync candidate count failed");
        check_cuda(cudaStreamSynchronize(impl_->stream), "cudaStreamSynchronize profiled scan failed");

        count = std::min<unsigned int>(count, static_cast<unsigned int>(kMaxCandidates));
        std::vector<Candidate> out;
        if (count != 0) {
            std::vector<DeviceCandidate> raw(count);
            check_cuda(cudaMemcpy(raw.data(), impl_->d_candidates,
                                  count * sizeof(DeviceCandidate), cudaMemcpyDeviceToHost),
                       "cudaMemcpy candidates failed");
            out.reserve(count);
            for (const auto& item : raw) {
                Candidate candidate;
                candidate.nonce = item.nonce;
                std::copy(std::begin(item.hash), std::end(item.hash), candidate.hash.begin());
                out.push_back(candidate);
            }
        }

        cudaEventDestroy(section_stop);
        cudaEventDestroy(section_start);
        cudaEventDestroy(total_stop);
        cudaEventDestroy(total_start);
        return out;
    } catch (...) {
        cudaEventDestroy(section_stop);
        cudaEventDestroy(section_start);
        cudaEventDestroy(total_stop);
        cudaEventDestroy(total_start);
        throw;
    }
}

int BatchEngine::device_id() const noexcept { return impl_->device_id; }
std::size_t BatchEngine::batch_size() const noexcept { return impl_->batch_size; }
bool BatchEngine::hash_pipeline_ready() const noexcept
{
    return full_ghostrider_cuda_coverage();
}

Hash512 keccak512_reference_stage(int device_id, const std::uint8_t* input, std::size_t length)
{
    return run_validation_kernel(device_id, input, length, true);
}

Hash512 cubehash512_reference_stage(int device_id, const std::uint8_t* input, std::size_t length)
{
    return run_validation_kernel(device_id, input, length, false);
}

int device_count()
{
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver) {
        cudaGetLastError();
        return 0;
    }
    check_cuda(status, "cudaGetDeviceCount failed");
    return count;
}

std::vector<DeviceInfo> enumerate_devices()
{
    std::vector<DeviceInfo> devices;
    const int count = device_count();
    devices.reserve(static_cast<std::size_t>(count));
    for (int device = 0; device < count; ++device) {
        cudaDeviceProp props{};
        check_cuda(cudaGetDeviceProperties(&props, device), "cudaGetDeviceProperties failed");
        devices.push_back(DeviceInfo{device, props.major, props.minor,
                                     props.totalGlobalMem, props.multiProcessorCount,
                                     props.warpSize});
    }
    return devices;
}

void print_devices()
{
    for (const auto& info : enumerate_devices()) {
        cudaDeviceProp props{};
        check_cuda(cudaGetDeviceProperties(&props, info.id), "cudaGetDeviceProperties failed");
        const double memory_gib = static_cast<double>(info.total_memory) /
                                  (1024.0 * 1024.0 * 1024.0);
        std::cout << "GPU " << info.id << ": " << props.name
                  << " | CC " << info.compute_major << '.' << info.compute_minor
                  << " | SMs " << info.multiprocessors
                  << " | warp " << info.warp_size
                  << " | " << memory_gib << " GiB\n";
    }
}

} // namespace yerbas::cuda

int yerbas_cuda_device_count() { return yerbas::cuda::device_count(); }
void yerbas_cuda_print_devices() { yerbas::cuda::print_devices(); }
