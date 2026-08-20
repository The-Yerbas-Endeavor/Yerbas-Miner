#include "cuda/cryptonight/cn_validation.h"
#include "cuda/cryptonight/cn_config.cuh"
#include "cuda/cryptonight/cn_final.cuh"
#include "cuda/cryptonight/cn_slow_hash.cuh"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace yerbas::cuda::cryptonight {
namespace {

void check(cudaError_t status, const char* what)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

__global__ void validation_kernel(std::uint8_t variant,
                                  const std::uint8_t* input,
                                  std::size_t length,
                                  std::uint8_t* scratchpad,
                                  std::uint8_t* output,
                                  int* ok)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const bool success = slow_hash(variant, input, length, scratchpad, output);
    *ok = success ? 1 : 0;
}

} // namespace

Hash256 validation_hash(int device_id,
                        std::uint8_t variant,
                        const std::uint8_t* input,
                        std::size_t length,
                        float* kernel_ms)
{
    const VariantConfig* cfg = config(variant);
    if (cfg == nullptr) throw std::invalid_argument("invalid CryptoNight CUDA variant");
    if (input == nullptr || length < 43)
        throw std::invalid_argument("CryptoNight validation input must be at least 43 bytes");

    check(cudaSetDevice(device_id), "cudaSetDevice failed");
    std::uint8_t* d_input = nullptr;
    std::uint8_t* d_scratchpad = nullptr;
    std::uint8_t* d_output = nullptr;
    int* d_ok = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;

    try {
        check(cudaMalloc(reinterpret_cast<void**>(&d_input), length), "cudaMalloc CN input failed");
        check(cudaMalloc(reinterpret_cast<void**>(&d_scratchpad), cfg->page_size), "cudaMalloc CN scratchpad failed");
        check(cudaMalloc(reinterpret_cast<void**>(&d_output), 32), "cudaMalloc CN output failed");
        check(cudaMalloc(reinterpret_cast<void**>(&d_ok), sizeof(int)), "cudaMalloc CN status failed");
        check(cudaMemcpy(d_input, input, length, cudaMemcpyHostToDevice), "cudaMemcpy CN input failed");
        check(cudaMemset(d_ok, 0, sizeof(int)), "cudaMemset CN status failed");

        if (kernel_ms != nullptr) {
            check(cudaEventCreate(&start_event), "cudaEventCreate start failed");
            check(cudaEventCreate(&stop_event), "cudaEventCreate stop failed");
            check(cudaEventRecord(start_event), "cudaEventRecord start failed");
        }

        validation_kernel<<<1, 1>>>(variant, d_input, length, d_scratchpad, d_output, d_ok);
        check(cudaGetLastError(), "CryptoNight validation kernel launch failed");

        if (kernel_ms != nullptr) {
            check(cudaEventRecord(stop_event), "cudaEventRecord stop failed");
            check(cudaEventSynchronize(stop_event), "cudaEventSynchronize stop failed");
            check(cudaEventElapsedTime(kernel_ms, start_event, stop_event), "cudaEventElapsedTime failed");
        } else {
            check(cudaDeviceSynchronize(), "CryptoNight validation kernel failed");
        }

        int ok = 0;
        check(cudaMemcpy(&ok, d_ok, sizeof(ok), cudaMemcpyDeviceToHost), "cudaMemcpy CN status failed");
        if (!ok) throw std::runtime_error("CryptoNight validation kernel rejected variant/input");

        Hash256 out{};
        check(cudaMemcpy(out.data(), d_output, out.size(), cudaMemcpyDeviceToHost), "cudaMemcpy CN output failed");

        if (stop_event) cudaEventDestroy(stop_event);
        if (start_event) cudaEventDestroy(start_event);
        cudaFree(d_ok); cudaFree(d_output); cudaFree(d_scratchpad); cudaFree(d_input);
        return out;
    } catch (...) {
        if (stop_event) cudaEventDestroy(stop_event);
        if (start_event) cudaEventDestroy(start_event);
        if (d_ok) cudaFree(d_ok);
        if (d_output) cudaFree(d_output);
        if (d_scratchpad) cudaFree(d_scratchpad);
        if (d_input) cudaFree(d_input);
        throw;
    }
}

} // namespace yerbas::cuda::cryptonight
