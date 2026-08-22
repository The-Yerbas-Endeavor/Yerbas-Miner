#include "cuda/cuda_backend.h"
#include "cuda/core/stage_dispatch.cuh"
#include "cuda/cryptonight/cn_config.cuh"
#include "cuda/cryptonight/cn_split.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace yerbas::cuda {
namespace {
void check_cuda_batch(cudaError_t status, const char* what) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}
__global__ void validate_kernel(std::uint8_t algorithm, const std::uint8_t* input,
                                std::size_t length, std::uint8_t* output, int* ok) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    std::uint8_t digest[64];
    const bool valid = core::dispatch_core512(algorithm, input, length, digest);
    *ok = valid ? 1 : 0;
    if (valid) for (int i = 0; i < 64; ++i) output[i] = digest[i];
}
Hash512 validate_stage(int device_id, std::uint8_t algorithm,
                       const std::uint8_t* input, std::size_t length) {
    if (!input || length == 0) throw std::invalid_argument("CUDA validation input must not be empty");
    check_cuda_batch(cudaSetDevice(device_id), "cudaSetDevice failed");
    std::uint8_t *d_input = nullptr, *d_output = nullptr;
    int* d_ok = nullptr;
    check_cuda_batch(cudaMalloc(reinterpret_cast<void**>(&d_input), length), "cudaMalloc input failed");
    try {
        check_cuda_batch(cudaMalloc(reinterpret_cast<void**>(&d_output), 64), "cudaMalloc output failed");
        check_cuda_batch(cudaMalloc(reinterpret_cast<void**>(&d_ok), sizeof(int)), "cudaMalloc status failed");
        check_cuda_batch(cudaMemcpy(d_input, input, length, cudaMemcpyHostToDevice), "cudaMemcpy input failed");
        validate_kernel<<<1,1>>>(algorithm, d_input, length, d_output, d_ok);
        check_cuda_batch(cudaGetLastError(), "validation kernel launch failed");
        check_cuda_batch(cudaDeviceSynchronize(), "validation synchronize failed");
        int ok = 0;
        check_cuda_batch(cudaMemcpy(&ok, d_ok, sizeof(ok), cudaMemcpyDeviceToHost), "cudaMemcpy status failed");
        if (!ok) throw std::runtime_error("CUDA validation dispatcher rejected algorithm");
        Hash512 out{};
        check_cuda_batch(cudaMemcpy(out.data(), d_output, out.size(), cudaMemcpyDeviceToHost), "cudaMemcpy output failed");
        cudaFree(d_ok); cudaFree(d_output); cudaFree(d_input);
        return out;
    } catch (...) {
        if (d_ok) cudaFree(d_ok); if (d_output) cudaFree(d_output); if (d_input) cudaFree(d_input); throw;
    }
}

template <std::uint8_t Variant>
__global__ void validate_cn_kernel(const std::uint8_t* input,
                                   std::uint8_t* scratchpad,
                                   std::uint8_t* output) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    cryptonight::SplitContext context{};
    std::uint8_t digest[32];
    cryptonight::split_setup<Variant>(input, 64, scratchpad, context);
    cryptonight::split_memory_loop<Variant>(scratchpad, context);
    cryptonight::split_finalize<Variant>(scratchpad, context, digest);
    for (int i = 0; i < 32; ++i) output[i] = digest[i];
    for (int i = 32; i < 64; ++i) output[i] = 0;
}

template <std::uint8_t Variant>
Hash512 validate_cn_stage(int device_id, const std::uint8_t* input, std::size_t length) {
    if (!input || length != 64) throw std::invalid_argument("CUDA CryptoNight validation input must be exactly 64 bytes");
    constexpr auto cfg = cryptonight::config_value(Variant);
    check_cuda_batch(cudaSetDevice(device_id), "cudaSetDevice failed");
    std::uint8_t *d_input = nullptr, *d_scratchpad = nullptr, *d_output = nullptr;
    check_cuda_batch(cudaMalloc(reinterpret_cast<void**>(&d_input), 64), "cudaMalloc CN input failed");
    try {
        check_cuda_batch(cudaMalloc(reinterpret_cast<void**>(&d_scratchpad), static_cast<std::size_t>(cfg.page_size)),
                         "cudaMalloc CN scratchpad failed");
        check_cuda_batch(cudaMalloc(reinterpret_cast<void**>(&d_output), 64), "cudaMalloc CN output failed");
        check_cuda_batch(cudaMemcpy(d_input, input, 64, cudaMemcpyHostToDevice), "cudaMemcpy CN input failed");
        validate_cn_kernel<Variant><<<1,1>>>(d_input, d_scratchpad, d_output);
        check_cuda_batch(cudaGetLastError(), "CN validation kernel launch failed");
        check_cuda_batch(cudaDeviceSynchronize(), "CN validation synchronize failed");
        Hash512 out{};
        check_cuda_batch(cudaMemcpy(out.data(), d_output, out.size(), cudaMemcpyDeviceToHost), "cudaMemcpy CN output failed");
        cudaFree(d_output); cudaFree(d_scratchpad); cudaFree(d_input);
        return out;
    } catch (...) {
        if (d_output) cudaFree(d_output);
        if (d_scratchpad) cudaFree(d_scratchpad);
        if (d_input) cudaFree(d_input);
        throw;
    }
}
}
Hash512 bmw512_reference_stage(int d, const std::uint8_t* p, std::size_t n) { return validate_stage(d,1,p,n); }
Hash512 groestl512_reference_stage(int d, const std::uint8_t* p, std::size_t n) { return validate_stage(d,2,p,n); }
Hash512 jh512_reference_stage(int d, const std::uint8_t* p, std::size_t n) { return validate_stage(d,3,p,n); }
Hash512 luffa512_reference_stage(int d, const std::uint8_t* p, std::size_t n) { return validate_stage(d,6,p,n); }
Hash512 shavite512_reference_stage(int d, const std::uint8_t* p, std::size_t n) { return validate_stage(d,8,p,n); }
Hash512 simd512_reference_stage(int d, const std::uint8_t* p, std::size_t n) { return validate_stage(d,9,p,n); }
Hash512 echo512_reference_stage(int d, const std::uint8_t* p, std::size_t n) { return validate_stage(d,10,p,n); }
Hash512 hamsi512_reference_stage(int d, const std::uint8_t* p, std::size_t n) { return validate_stage(d,11,p,n); }
Hash512 fugue512_reference_stage(int d, const std::uint8_t* p, std::size_t n) { return validate_stage(d,12,p,n); }
Hash512 shabal512_reference_stage(int d, const std::uint8_t* p, std::size_t n) { return validate_stage(d,13,p,n); }
Hash512 whirlpool512_reference_stage(int d, const std::uint8_t* p, std::size_t n) { return validate_stage(d,14,p,n); }
Hash512 cryptonight_reference_stage(int d, const std::uint8_t* p, std::size_t n, std::uint8_t variant) {
    switch (variant) {
    case 0: return validate_cn_stage<0>(d,p,n);
    case 1: return validate_cn_stage<1>(d,p,n);
    case 2: return validate_cn_stage<2>(d,p,n);
    case 3: return validate_cn_stage<3>(d,p,n);
    case 4: return validate_cn_stage<4>(d,p,n);
    case 5: return validate_cn_stage<5>(d,p,n);
    default: throw std::invalid_argument("CUDA CryptoNight variant must be 0..5");
    }
}
}
