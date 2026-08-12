#include "cuda/cuda_backend.h"
#include "cuda/core/keccak512.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace yerbas::cuda {
namespace {

constexpr std::size_t kMaxCandidates = 64;
constexpr std::size_t kStateBytes = 64;
constexpr std::uint8_t kKeccakCoreIndex = 4;

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

// First real GhostRider core stage: Keccak-512 (Yerbas coreHash index 4).
// stage_index == 0 hashes the 80-byte block header with the per-thread nonce.
// Later stages hash the previous 64-byte state in-place.
__global__ void keccak512_stage(const DeviceJob* job,
                                const std::uint32_t* nonces,
                                std::uint8_t* states,
                                std::size_t count,
                                int stage_index)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;

    std::uint8_t digest[64];

    if (stage_index == 0) {
        std::uint8_t header[80];
        #pragma unroll
        for (int i = 0; i < 80; ++i) header[i] = job->header[i];

        const std::uint32_t nonce = nonces[index];
        header[76] = static_cast<std::uint8_t>(nonce);
        header[77] = static_cast<std::uint8_t>(nonce >> 8);
        header[78] = static_cast<std::uint8_t>(nonce >> 16);
        header[79] = static_cast<std::uint8_t>(nonce >> 24);
        core::keccak512(header, 80, digest);
    } else {
        const std::uint8_t* input = states + index * kStateBytes;
        core::keccak512(input, kStateBytes, digest);
    }

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

// Once the final 512-bit GhostRider state exists, only its low 256 bits are
// compared with the share target, matching Yerbas Core's trim256().
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
    for (int i = 0; i < 32; ++i) {
        candidates[slot].hash[i] = hash[i];
    }
}

} // namespace

struct BatchEngine::Impl {
    int device_id{-1};
    std::size_t batch_size{0};
    cudaStream_t stream{};
    DeviceJob* d_job{nullptr};
    std::uint32_t* d_nonces{nullptr};
    std::uint8_t* d_states{nullptr};
    DeviceCandidate* d_candidates{nullptr};
    unsigned int* d_candidate_count{nullptr};
    bool job_loaded{false};

    Impl(int id, std::size_t size)
        : device_id(id), batch_size(size)
    {
        if (batch_size == 0) {
            throw std::runtime_error("CUDA batch size must be greater than zero");
        }

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

BatchEngine::BatchEngine(int device_id, std::size_t batch_size)
    : impl_(std::make_unique<Impl>(device_id, batch_size))
{
}

BatchEngine::~BatchEngine() = default;
BatchEngine::BatchEngine(BatchEngine&&) noexcept = default;
BatchEngine& BatchEngine::operator=(BatchEngine&&) noexcept = default;

void BatchEngine::upload_job(const JobDescriptor& job)
{
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
    if (!impl_->job_loaded) {
        throw std::runtime_error("CUDA batch scan requested before upload_job");
    }

    if (!hash_pipeline_ready()) {
        // One core kernel is now implemented and independently testable, but a
        // complete GhostRider share requires all 15 core hashes plus the three
        // selected CryptoNight stages. Never submit partial-pipeline results.
        return {};
    }

    check_cuda(cudaSetDevice(impl_->device_id), "cudaSetDevice failed");
    check_cuda(cudaMemsetAsync(impl_->d_candidate_count, 0, sizeof(unsigned int), impl_->stream),
               "cudaMemsetAsync candidate counter failed");

    constexpr int threads = 256;
    const int blocks = static_cast<int>((impl_->batch_size + threads - 1) / threads);
    initialize_nonce_batch<<<blocks, threads, 0, impl_->stream>>>(start_nonce,
                                                                  impl_->d_nonces,
                                                                  impl_->batch_size);
    check_cuda(cudaGetLastError(), "initialize_nonce_batch launch failed");

    // Dispatcher will be enabled only after every possible job-selected stage
    // has a validated CUDA implementation. Keccak-512 is implemented today as
    // the first core stage and exercised through keccak512_reference_stage().

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
bool BatchEngine::hash_pipeline_ready() const noexcept { return false; }

Hash512 keccak512_reference_stage(int device_id,
                                  const std::uint8_t* input,
                                  std::size_t length)
{
    if (input == nullptr || length == 0) {
        throw std::invalid_argument("CUDA Keccak validation input must not be empty");
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

        keccak512_validation_kernel<<<1, 1>>>(d_input, length, d_output);
        check_cuda(cudaGetLastError(), "keccak512 validation launch failed");
        check_cuda(cudaDeviceSynchronize(), "keccak512 validation synchronize failed");

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
