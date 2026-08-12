#include "cuda/cuda_backend.h"
#include "cuda/core/blake512.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace yerbas::cuda {
namespace {

void check_cuda_blake(cudaError_t status, const char* what)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

__global__ void blake512_validation_kernel(const std::uint8_t* input,
                                           std::size_t length,
                                           std::uint8_t* output)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    std::uint8_t digest[64];
    core::blake512(input, length, digest);
    #pragma unroll
    for (int i = 0; i < 64; ++i) output[i] = digest[i];
}

} // namespace

Hash512 blake512_reference_stage(int device_id,
                                 const std::uint8_t* input,
                                 std::size_t length)
{
    if (input == nullptr || length == 0 || length > 111) {
        throw std::invalid_argument("CUDA BLAKE-512 validation expects 1..111 input bytes");
    }

    check_cuda_blake(cudaSetDevice(device_id), "cudaSetDevice failed");
    std::uint8_t* d_input = nullptr;
    std::uint8_t* d_output = nullptr;

    check_cuda_blake(cudaMalloc(reinterpret_cast<void**>(&d_input), length),
                     "cudaMalloc BLAKE input failed");
    try {
        check_cuda_blake(cudaMalloc(reinterpret_cast<void**>(&d_output), 64),
                         "cudaMalloc BLAKE output failed");
        check_cuda_blake(cudaMemcpy(d_input, input, length, cudaMemcpyHostToDevice),
                         "cudaMemcpy BLAKE input failed");

        blake512_validation_kernel<<<1, 1>>>(d_input, length, d_output);
        check_cuda_blake(cudaGetLastError(), "BLAKE-512 validation launch failed");
        check_cuda_blake(cudaDeviceSynchronize(), "BLAKE-512 validation synchronize failed");

        Hash512 out{};
        check_cuda_blake(cudaMemcpy(out.data(), d_output, out.size(), cudaMemcpyDeviceToHost),
                         "cudaMemcpy BLAKE output failed");
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
