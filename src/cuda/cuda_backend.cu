#include "cuda/cuda_backend.h"
#include "cuda/core/stage_dispatch.cuh"
#include "cuda/core_coverage.h"
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
// Until all 18 selectable stages are native CUDA, keep fallback batches small.
// This makes the bootstrap path useful for correctness/accepted-share testing
// without spending seconds copying and reference-hashing a 65K state batch.
constexpr std::size_t kBootstrapBatch = 256;

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
    DeviceCandidate* d_candidates{nullptr};
    unsigned int* d_candidate_count{nullptr};
    JobDescriptor host_job{};
    std::vector<std::uint8_t> host_states;
    bool job_loaded{false};

    Impl(int id, std::size_t requested_size, unsigned int requested_fallback_threads)
        : device_id(id),
          batch_size(full_ghostrider_cuda_coverage()
                         ? requested_size
                         : std::min(requested_size, kBootstrapBatch)),
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
    if (!hash_pipeline_ready()) return {};

    check_cuda(cudaSetDevice(impl_->device_id), "cudaSetDevice failed");
    check_cuda(cudaMemsetAsync(impl_->d_candidate_count, 0, sizeof(unsigned int), impl_->stream),
               "cudaMemsetAsync candidate counter failed");

    constexpr int threads = 256;
    const int blocks = static_cast<int>((impl_->batch_size + threads - 1) / threads);
    initialize_nonce_batch<<<blocks, threads, 0, impl_->stream>>>(start_nonce,
                                                                  impl_->d_nonces,
                                                                  impl_->batch_size);
    check_cuda(cudaGetLastError(), "initialize_nonce_batch launch failed");

    // Bootstrap dispatcher: validated CUDA cores run on-device. Missing cores
    // and CryptoNight variants run through the exact pinned Yerbas Core stage
    // implementation on the host. As native kernels land, the fallback count
    // shrinks without changing the final GhostRider result.
    for (int stage_index = 0; stage_index < 18; ++stage_index) {
        const std::uint8_t stage = impl_->host_job.stages[static_cast<std::size_t>(stage_index)];
        const bool is_cn = (stage & ghostrider::kCryptoNightStageFlag) != 0;
        const std::uint8_t algorithm = static_cast<std::uint8_t>(stage & 0x7fU);
        const bool native_cuda = !is_cn && core::core512_implemented(algorithm);

        if (native_cuda) {
            core512_stage<<<blocks, threads, 0, impl_->stream>>>(impl_->d_job,
                                                                 impl_->d_nonces,
                                                                 impl_->d_states,
                                                                 impl_->batch_size,
                                                                 stage_index,
                                                                 algorithm);
            check_cuda(cudaGetLastError(), "GhostRider CUDA core stage launch failed");
            continue;
        }

        if (stage_index > 0) {
            check_cuda(cudaMemcpyAsync(impl_->host_states.data(), impl_->d_states,
                                       impl_->host_states.size(), cudaMemcpyDeviceToHost,
                                       impl_->stream),
                       "cudaMemcpyAsync fallback states failed");
            check_cuda(cudaStreamSynchronize(impl_->stream),
                       "cudaStreamSynchronize fallback read failed");
        }

        const std::size_t worker_count = std::min<std::size_t>(impl_->fallback_threads,
                                                                impl_->batch_size);
        const std::size_t chunk = (impl_->batch_size + worker_count - 1) / worker_count;
        std::vector<std::thread> fallback_workers;
        fallback_workers.reserve(worker_count);

        for (std::size_t worker = 0; worker < worker_count; ++worker) {
            const std::size_t begin = worker * chunk;
            const std::size_t end = std::min(impl_->batch_size, begin + chunk);
            if (begin >= end) break;

            fallback_workers.emplace_back([&, begin, end, stage_index, stage, start_nonce]() {
                for (std::size_t i = begin; i < end; ++i) {
                    ghostrider::Hash512 digest{};
                    if (stage_index == 0) {
                        std::array<std::uint8_t, 80> header = impl_->host_job.header;
                        const std::uint32_t nonce = start_nonce + static_cast<std::uint32_t>(i);
                        header[76] = static_cast<std::uint8_t>(nonce);
                        header[77] = static_cast<std::uint8_t>(nonce >> 8);
                        header[78] = static_cast<std::uint8_t>(nonce >> 16);
                        header[79] = static_cast<std::uint8_t>(nonce >> 24);
                        digest = ghostrider::stage_reference({header.data(), header.size()}, stage);
                    } else {
                        std::uint8_t* state = impl_->host_states.data() + i * kStateBytes;
                        digest = ghostrider::stage_reference({state, kStateBytes}, stage);
                    }
                    std::memcpy(impl_->host_states.data() + i * kStateBytes,
                                digest.data(), kStateBytes);
                }
            });
        }

        for (auto& worker : fallback_workers) worker.join();

        check_cuda(cudaMemcpyAsync(impl_->d_states, impl_->host_states.data(),
                                   impl_->host_states.size(), cudaMemcpyHostToDevice,
                                   impl_->stream),
                   "cudaMemcpyAsync fallback states upload failed");
    }

    collect_candidates<<<blocks, threads, 0, impl_->stream>>>(impl_->d_nonces,
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

int BatchEngine::device_id() const noexcept { return impl_->device_id; }
std::size_t BatchEngine::batch_size() const noexcept { return impl_->batch_size; }
bool BatchEngine::hash_pipeline_ready() const noexcept
{
    // Native full-CUDA coverage is still the performance goal, but the hybrid
    // stage fallback provides a complete, reference-correct mining pipeline.
    return ghostrider::reference_ready();
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
