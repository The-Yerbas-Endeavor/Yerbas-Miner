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

__global__ void keccak_prefix_kernel(const std::uint8_t* input,
                                     std::size_t length,
                                     std::uint8_t* output)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    std::uint8_t state[200];
    keccak1600(input, length, state);
    #pragma unroll
    for (int i = 0; i < 32; ++i) output[i] = state[i];
}

__global__ void checkpoint_kernel(std::uint8_t variant,
                                  const std::uint8_t* input,
                                  std::size_t length,
                                  std::uint8_t* scratchpad,
                                  ValidationCheckpoints* output,
                                  int* ok)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (variant >= 6 || input == nullptr || length < 43 || scratchpad == nullptr || output == nullptr) {
        *ok = 0;
        return;
    }

    const VariantConfig cfg = config_value(variant);
    std::uint8_t state[200];
    keccak1600(input, length, state);

    std::uint8_t text[128];
    #pragma unroll
    for (int i = 0; i < 128; ++i) text[i] = state[64 + i];

    std::uint8_t expanded[240];
    aes256_expand_key(state, expanded);
    #pragma unroll
    for (int i = 0; i < 64; ++i) output->expanded_key_prefix[i] = expanded[i];

    const std::size_t init_rounds = cfg.page_size / 128U;
    for (std::size_t i = 0; i < init_rounds; ++i) {
        #pragma unroll
        for (int block = 0; block < 8; ++block)
            aes_pseudo_round(text + block * 16, expanded);
        for (int b = 0; b < 128; ++b)
            scratchpad[i * 128U + static_cast<std::size_t>(b)] = text[b];
    }
    #pragma unroll
    for (int i = 0; i < 128; ++i) output->scratchpad_prefix[i] = scratchpad[i];

    std::uint8_t a[16], b[16], c[16], t[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        a[i] = static_cast<std::uint8_t>(state[i] ^ state[32 + i]);
        b[i] = static_cast<std::uint8_t>(state[16 + i] ^ state[48 + i]);
    }

    const std::uint64_t tweak = cn_load64(input + 35) ^ cn_load64(state + 192);

    std::size_t j = cn_index(a, cfg.aes_rounds);
    std::uint8_t* slot = scratchpad + j * 16U;
    aes_single_round(slot, c, a);
    #pragma unroll
    for (int k = 0; k < 16; ++k) slot[k] = static_cast<std::uint8_t>(c[k] ^ b[k]);
    variant1_mutate(slot);

    j = cn_index(c, cfg.aes_rounds);
    slot = scratchpad + j * 16U;
    copy16(t, slot);

    const std::uint64_t c0 = cn_load64(c);
    const std::uint64_t t0 = cn_load64(t);
    const std::uint64_t lo = c0 * t0;
    const std::uint64_t hi = __umul64hi(c0, t0);

    std::uint64_t a0 = cn_load64(a) + hi;
    std::uint64_t a1 = cn_load64(a + 8) + lo;
    cn_store64(slot, a0);
    cn_store64(slot + 8, a1 ^ tweak);

    a0 ^= t0;
    a1 ^= cn_load64(t + 8);
    cn_store64(a, a0);
    cn_store64(a + 8, a1);
    copy16(b, c);

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        output->first_loop_state[i] = a[i];
        output->first_loop_state[16 + i] = b[i];
        output->first_loop_state[32 + i] = c[i];
        output->first_loop_state[48 + i] = t[i];
    }
    *ok = 1;
}

} // namespace

Hash256 validation_keccak_prefix(int device_id,
                                 const std::uint8_t* input,
                                 std::size_t length)
{
    if (input == nullptr || length == 0)
        throw std::invalid_argument("CryptoNight Keccak validation input must not be empty");

    check(cudaSetDevice(device_id), "cudaSetDevice failed");
    std::uint8_t* d_input = nullptr;
    std::uint8_t* d_output = nullptr;

    try {
        check(cudaMalloc(reinterpret_cast<void**>(&d_input), length), "cudaMalloc CN Keccak input failed");
        check(cudaMalloc(reinterpret_cast<void**>(&d_output), 32), "cudaMalloc CN Keccak output failed");
        check(cudaMemcpy(d_input, input, length, cudaMemcpyHostToDevice), "cudaMemcpy CN Keccak input failed");

        keccak_prefix_kernel<<<1, 1>>>(d_input, length, d_output);
        check(cudaGetLastError(), "CryptoNight Keccak checkpoint kernel launch failed");
        check(cudaDeviceSynchronize(), "CryptoNight Keccak checkpoint kernel failed");

        Hash256 out{};
        check(cudaMemcpy(out.data(), d_output, out.size(), cudaMemcpyDeviceToHost),
              "cudaMemcpy CN Keccak checkpoint failed");

        cudaFree(d_output);
        cudaFree(d_input);
        return out;
    } catch (...) {
        if (d_output) cudaFree(d_output);
        if (d_input) cudaFree(d_input);
        throw;
    }
}

ValidationCheckpoints validation_checkpoints(int device_id,
                                             std::uint8_t variant,
                                             const std::uint8_t* input,
                                             std::size_t length)
{
    const VariantConfig* cfg = config(variant);
    if (cfg == nullptr) throw std::invalid_argument("invalid CryptoNight CUDA variant");
    if (input == nullptr || length < 43)
        throw std::invalid_argument("CryptoNight checkpoint input must be at least 43 bytes");

    check(cudaSetDevice(device_id), "cudaSetDevice failed");
    std::uint8_t* d_input = nullptr;
    std::uint8_t* d_scratchpad = nullptr;
    ValidationCheckpoints* d_output = nullptr;
    int* d_ok = nullptr;

    try {
        check(cudaMalloc(reinterpret_cast<void**>(&d_input), length), "cudaMalloc CN checkpoint input failed");
        check(cudaMalloc(reinterpret_cast<void**>(&d_scratchpad), cfg->page_size), "cudaMalloc CN checkpoint scratchpad failed");
        check(cudaMalloc(reinterpret_cast<void**>(&d_output), sizeof(ValidationCheckpoints)), "cudaMalloc CN checkpoint output failed");
        check(cudaMalloc(reinterpret_cast<void**>(&d_ok), sizeof(int)), "cudaMalloc CN checkpoint status failed");
        check(cudaMemcpy(d_input, input, length, cudaMemcpyHostToDevice), "cudaMemcpy CN checkpoint input failed");
        check(cudaMemset(d_ok, 0, sizeof(int)), "cudaMemset CN checkpoint status failed");

        checkpoint_kernel<<<1, 1>>>(variant, d_input, length, d_scratchpad, d_output, d_ok);
        check(cudaGetLastError(), "CryptoNight checkpoint kernel launch failed");
        check(cudaDeviceSynchronize(), "CryptoNight checkpoint kernel failed");

        int ok = 0;
        check(cudaMemcpy(&ok, d_ok, sizeof(ok), cudaMemcpyDeviceToHost), "cudaMemcpy CN checkpoint status failed");
        if (!ok) throw std::runtime_error("CryptoNight checkpoint kernel rejected variant/input");

        ValidationCheckpoints out{};
        check(cudaMemcpy(&out, d_output, sizeof(out), cudaMemcpyDeviceToHost), "cudaMemcpy CN checkpoints failed");

        cudaFree(d_ok); cudaFree(d_output); cudaFree(d_scratchpad); cudaFree(d_input);
        return out;
    } catch (...) {
        if (d_ok) cudaFree(d_ok);
        if (d_output) cudaFree(d_output);
        if (d_scratchpad) cudaFree(d_scratchpad);
        if (d_input) cudaFree(d_input);
        throw;
    }
}

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
