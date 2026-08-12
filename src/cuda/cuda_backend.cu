#include "cuda/cuda_backend.h"

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

// This kernel is intentionally separated from the future GhostRider stage
// kernels. Once a batch of final 256-bit hashes exists in d_hashes, target
// comparison happens entirely on-device and only successful candidates cross
// PCIe back to the host.
__global__ void collect_candidates(const std::uint32_t* nonces,
                                   const std::uint8_t* hashes,
                                   std::size_t count,
                                   const DeviceJob* job,
                                   DeviceCandidate* candidates,
                                   unsigned int* candidate_count)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;

    const std::uint8_t* hash = hashes + index * 32;
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
    std::uint8_t* d_hashes{nullptr};
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
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_hashes), batch_size * 32),
                   "cudaMalloc hash batch failed");
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
        if (d_hashes) cudaFree(d_hashes);
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
        // The persistent batch engine is live, but the GhostRider stage kernels
        // are not yet installed. Returning no candidates is safer than treating
        // uninitialized d_hashes as valid PoW results.
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

    // Future implementation point:
    //   dispatch the 18 job-selected GhostRider stages here. The selector is
    //   already resident in d_job->stages, so kernels never need to recompute
    //   selection per nonce. d_hashes remains resident for the whole batch.

    collect_candidates<<<blocks, threads, 0, impl_->stream>>>(impl_->d_nonces,
                                                               impl_->d_hashes,
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
