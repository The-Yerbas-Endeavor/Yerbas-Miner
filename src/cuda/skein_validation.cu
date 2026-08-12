#include "cuda/cuda_backend.h"
#include "cuda/core/skein512.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace yerbas::cuda {
namespace {

void check_cuda_skein(cudaError_t status, const char* what)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

__global__ void skein512_validation_kernel(const std::uint8_t* input,
                                           std::size_t length,
                                           std::uint8_t* output)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    std::uint8_t digest[64];
    core::skein512(input, length, digest);
    #pragma unroll
    for (int i = 0; i < 64; ++i) output[i] = digest[i];
}

} // namespace

Hash512 skein512_reference_stage(int device_id,
                                 const std::uint8_t* input,
                                 std::size_t length)
{
    if (input == nullptr || (length != 64 && length != 80)) {
        throw std::invalid_argument("CUDA Skein-512 validation expects a 64- or 80-byte input");
    }

    check_cuda_skein(cudaSetDevice(device_id), "cudaSetDevice failed");
    std::uint8_t* d_input = nullptr;
    std::uint8_t* d_output = nullptr;

    check_cuda_skein(cudaMalloc(reinterpret_cast<void**>(&d_input), length),
                     "cudaMalloc Skein input failed");
    try {
        check_cuda_skein(cudaMalloc(reinterpret_cast<void**>(&d_output), 64),
                         "cudaMalloc Skein output failed");
        check_cuda_skein(cudaMemcpy(d_input, input, length, cudaMemcpyHostToDevice),
                         "cudaMemcpy Skein input failed");

        skein512_validation_kernel<<<1, 1>>>(d_input, length, d_output);
        check_cuda_skein(cudaGetLastError(), "Skein-512 validation launch failed");
        check_cuda_skein(cudaDeviceSynchronize(), "Skein-512 validation synchronize failed");

        Hash512 out{};
        check_cuda_skein(cudaMemcpy(out.data(), d_output, out.size(), cudaMemcpyDeviceToHost),
                         "cudaMemcpy Skein output failed");
        cudaFree(d_output);
        cudaFree(d_input);
        return out;
    } catch (...) {
        if (d_output) cudaFree(d_output);
        if (d_input) cudaFree(d_input);
        throw;
    }
}

} // namespace yerbas::cuda
