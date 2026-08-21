#include "cuda/cryptonight/cn_validation.h"
#include "cuda/cryptonight/cn_config.cuh"
#include "cuda/cryptonight/cn_final.cuh"
#include "cuda/cryptonight/cn_slow_hash.cuh"

#include <cuda_runtime.h>

extern "C" {
#include "c_keccak.h"
#include "oaes_lib.h"
int aesb_single_round(const std::uint8_t* in, std::uint8_t* out, const std::uint8_t* expandedKey);
int aesb_pseudo_round(const std::uint8_t* in, std::uint8_t* out, const std::uint8_t* expandedKey);
}

#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace yerbas::cuda::cryptonight {
namespace {

void check(cudaError_t status, const char* what)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

__device__ __forceinline__ void checkpoint_state(std::uint8_t* dst,
                                                 const std::uint8_t a[16],
                                                 const std::uint8_t b[32],
                                                 const std::uint8_t c[16],
                                                 const std::uint8_t t[16])
{
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        dst[i] = a[i];
        dst[16 + i] = b[i];
        dst[32 + i] = b[16 + i];
        dst[48 + i] = c[i];
        dst[64 + i] = t[i];
    }
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

    auto* expanded_key_prefix = reinterpret_cast<std::uint8_t*>(&output->expanded_key_prefix);
    #pragma unroll
    for (int i = 0; i < 64; ++i) expanded_key_prefix[i] = expanded[i];

    const std::size_t init_rounds = cfg.page_size / 128U;
    for (std::size_t i = 0; i < init_rounds; ++i) {
        #pragma unroll
        for (int block = 0; block < 8; ++block)
            aes_pseudo_round(text + block * 16, expanded);
        for (int x = 0; x < 128; ++x)
            scratchpad[i * 128U + static_cast<std::size_t>(x)] = text[x];
    }

    auto* scratchpad_prefix = reinterpret_cast<std::uint8_t*>(&output->scratchpad_prefix);
    #pragma unroll
    for (int i = 0; i < 128; ++i) scratchpad_prefix[i] = scratchpad[i];

    std::uint8_t a[16], b[32], c[16], t[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        a[i] = static_cast<std::uint8_t>(state[i] ^ state[32 + i]);
        b[i] = static_cast<std::uint8_t>(state[16 + i] ^ state[48 + i]);
        b[16 + i] = b[i];
        c[i] = 0;
        t[i] = 0;
    }

    const std::uint64_t tweak = cn_load64(input + 35) ^ cn_load64(state + 192);

    for (std::uint32_t iteration = 0; iteration < cfg.iterations; ++iteration) {
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
        copy16(b + 16, b);
        copy16(b, c);

        std::uint8_t* snapshot = nullptr;
        if (iteration == 0)
            snapshot = reinterpret_cast<std::uint8_t*>(&output->first_loop_state);
        else if (iteration == 1)
            snapshot = reinterpret_cast<std::uint8_t*>(&output->second_loop_state);
        else if (iteration == 15)
            snapshot = reinterpret_cast<std::uint8_t*>(&output->loop16_state);
        else if (iteration == 1023)
            snapshot = reinterpret_cast<std::uint8_t*>(&output->loop1024_state);
        else if (iteration + 1U == cfg.iterations)
            snapshot = reinterpret_cast<std::uint8_t*>(&output->final_loop_state);

        if (snapshot != nullptr) checkpoint_state(snapshot, a, b, c, t);
    }

    #pragma unroll
    for (int i = 0; i < 128; ++i) text[i] = state[64 + i];
    aes256_expand_key(state + 32, expanded);
    for (std::size_t i = 0; i < init_rounds; ++i) {
        #pragma unroll
        for (int block = 0; block < 8; ++block) {
            std::uint8_t* x = text + block * 16;
            xor16(x, scratchpad + i * 128U + static_cast<std::size_t>(block * 16));
            aes_pseudo_round(x, expanded);
        }
    }

    auto* collapsed = reinterpret_cast<std::uint8_t*>(&output->collapsed_text);
    #pragma unroll
    for (int i = 0; i < 128; ++i) collapsed[i] = text[i];

    #pragma unroll
    for (int i = 0; i < 128; ++i) state[64 + i] = text[i];
    keccak_permute_state(state);

    auto* post_keccak = reinterpret_cast<std::uint8_t*>(&output->post_keccak_state);
    #pragma unroll
    for (int i = 0; i < 200; ++i) post_keccak[i] = state[i];

    output->extra_hash_selector = static_cast<std::uint8_t>(state[0] & 3U);
    auto* final_hash = reinterpret_cast<std::uint8_t*>(&output->final_extra_hash);
    dispatch_extra_hash(output->extra_hash_selector, state, final_hash);

    *ok = 1;
}

std::uint64_t host_load64(const std::uint8_t* p)
{
    std::uint64_t v = 0;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

void host_store64(std::uint8_t* p, std::uint64_t v)
{
    std::memcpy(p, &v, sizeof(v));
}

void host_variant1_mutate(std::uint8_t block[16])
{
    const std::uint8_t tmp = block[11];
    constexpr std::uint32_t table = 0x75310U;
    const std::uint8_t index = static_cast<std::uint8_t>((((tmp >> 3) & 6U) | (tmp & 1U)) << 1);
    block[11] = static_cast<std::uint8_t>(tmp ^ ((table >> index) & 0x30U));
}

void host_mul128(std::uint64_t a, std::uint64_t b, std::uint64_t& hi, std::uint64_t& lo)
{
#if defined(__SIZEOF_INT128__)
    const unsigned __int128 product = static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b);
    lo = static_cast<std::uint64_t>(product);
    hi = static_cast<std::uint64_t>(product >> 64);
#else
    const std::uint64_t a0 = static_cast<std::uint32_t>(a);
    const std::uint64_t a1 = a >> 32;
    const std::uint64_t b0 = static_cast<std::uint32_t>(b);
    const std::uint64_t b1 = b >> 32;
    const std::uint64_t p0 = a0 * b0;
    const std::uint64_t p1 = a0 * b1;
    const std::uint64_t p2 = a1 * b0;
    const std::uint64_t p3 = a1 * b1;
    const std::uint64_t middle = (p0 >> 32) + static_cast<std::uint32_t>(p1) + static_cast<std::uint32_t>(p2);
    lo = (middle << 32) | static_cast<std::uint32_t>(p0);
    hi = p3 + (p1 >> 32) + (p2 >> 32) + (middle >> 32);
#endif
}

void host_snapshot(std::array<std::uint8_t, 80>& dst,
                   const std::uint8_t a[16],
                   const std::uint8_t b[32],
                   const std::uint8_t c[16],
                   const std::uint8_t t[16])
{
    std::memcpy(dst.data(), a, 16);
    std::memcpy(dst.data() + 16, b, 32);
    std::memcpy(dst.data() + 48, c, 16);
    std::memcpy(dst.data() + 64, t, 16);
}

ValidationCheckpoints core_dark_checkpoints(const std::uint8_t* input, std::size_t length)
{
    constexpr std::size_t page_size = 524288U;
    constexpr std::uint32_t iterations = 131072U;
    constexpr std::size_t aes_rounds = 32768U;

    ValidationCheckpoints out{};
    std::array<std::uint8_t, 200> state{};
    ::keccak1600(input, static_cast<int>(length), state.data());

    OAES_CTX* raw_ctx = oaes_alloc();
    if (raw_ctx == nullptr) throw std::runtime_error("oaes_alloc failed for Core checkpoint reference");
    auto* ctx = reinterpret_cast<oaes_ctx*>(raw_ctx);

    if (oaes_key_import_data(raw_ctx, state.data(), 32) != OAES_RET_SUCCESS) {
        oaes_free(&raw_ctx);
        throw std::runtime_error("oaes_key_import_data failed for Core checkpoint reference");
    }
    std::memcpy(out.expanded_key_prefix.data(), ctx->key->exp_data, out.expanded_key_prefix.size());

    std::array<std::uint8_t, 128> text{};
    std::memcpy(text.data(), state.data() + 64, text.size());
    std::vector<std::uint8_t> scratchpad(page_size);
    const std::size_t init_rounds = page_size / 128U;
    for (std::size_t i = 0; i < init_rounds; ++i) {
        for (int block = 0; block < 8; ++block) {
            std::uint8_t* x = text.data() + block * 16;
            aesb_pseudo_round(x, x, ctx->key->exp_data);
        }
        std::memcpy(scratchpad.data() + i * 128U, text.data(), 128);
    }
    std::memcpy(out.scratchpad_prefix.data(), scratchpad.data(), out.scratchpad_prefix.size());

    std::uint8_t a[16], b[32]{}, c[16]{}, t[16]{};
    for (int i = 0; i < 16; ++i) {
        a[i] = static_cast<std::uint8_t>(state[i] ^ state[32 + i]);
        b[i] = static_cast<std::uint8_t>(state[16 + i] ^ state[48 + i]);
        b[16 + i] = b[i];
    }
    const std::uint64_t tweak = host_load64(input + 35) ^ host_load64(state.data() + 192);

    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        std::size_t j = static_cast<std::size_t>((host_load64(a) >> 4) & (aes_rounds - 1));
        std::uint8_t* slot = scratchpad.data() + j * 16U;

        aesb_single_round(slot, c, a);
        for (int k = 0; k < 16; ++k) slot[k] = static_cast<std::uint8_t>(c[k] ^ b[k]);
        host_variant1_mutate(slot);

        j = static_cast<std::size_t>((host_load64(c) >> 4) & (aes_rounds - 1));
        slot = scratchpad.data() + j * 16U;
        std::memcpy(t, slot, 16);

        std::uint64_t hi = 0, lo = 0;
        host_mul128(host_load64(c), host_load64(t), hi, lo);
        std::uint64_t a0 = host_load64(a) + hi;
        std::uint64_t a1 = host_load64(a + 8) + lo;
        host_store64(slot, a0);
        host_store64(slot + 8, a1 ^ tweak);
        a0 ^= host_load64(t);
        a1 ^= host_load64(t + 8);
        host_store64(a, a0);
        host_store64(a + 8, a1);
        std::memcpy(b + 16, b, 16);
        std::memcpy(b, c, 16);

        if (iteration == 0) host_snapshot(out.first_loop_state, a, b, c, t);
        else if (iteration == 1) host_snapshot(out.second_loop_state, a, b, c, t);
        else if (iteration == 15) host_snapshot(out.loop16_state, a, b, c, t);
        else if (iteration == 1023) host_snapshot(out.loop1024_state, a, b, c, t);
        else if (iteration + 1U == iterations) host_snapshot(out.final_loop_state, a, b, c, t);
    }

    std::memcpy(text.data(), state.data() + 64, 128);
    if (oaes_key_import_data(raw_ctx, state.data() + 32, 32) != OAES_RET_SUCCESS) {
        oaes_free(&raw_ctx);
        throw std::runtime_error("second oaes_key_import_data failed for Core checkpoint reference");
    }
    for (std::size_t i = 0; i < init_rounds; ++i) {
        for (int block = 0; block < 8; ++block) {
            std::uint8_t* x = text.data() + block * 16;
            const std::uint8_t* s = scratchpad.data() + i * 128U + static_cast<std::size_t>(block * 16);
            for (int k = 0; k < 16; ++k) x[k] ^= s[k];
            aesb_pseudo_round(x, x, ctx->key->exp_data);
        }
    }
    std::memcpy(out.collapsed_text.data(), text.data(), 128);

    std::memcpy(state.data() + 64, text.data(), 128);
    ::keccakf(reinterpret_cast<std::uint64_t*>(state.data()), 24);
    std::memcpy(out.post_keccak_state.data(), state.data(), out.post_keccak_state.size());
    out.extra_hash_selector = static_cast<std::uint8_t>(state[0] & 3U);

    oaes_free(&raw_ctx);
    return out;
}

template <typename T>
const char* first_mismatch_name(const ValidationCheckpoints& cpu,
                                const ValidationCheckpoints& gpu,
                                const T ValidationCheckpoints::* member,
                                const char* name)
{
    return cpu.*member == gpu.*member ? nullptr : name;
}

void report_dark_checkpoint_result(const ValidationCheckpoints& cpu,
                                   const ValidationCheckpoints& gpu)
{
    const char* mismatch = nullptr;
    if ((mismatch = first_mismatch_name(cpu, gpu, &ValidationCheckpoints::second_loop_state, "memory-loop iteration 2")) ||
        (mismatch = first_mismatch_name(cpu, gpu, &ValidationCheckpoints::loop16_state, "memory-loop iteration 16")) ||
        (mismatch = first_mismatch_name(cpu, gpu, &ValidationCheckpoints::loop1024_state, "memory-loop iteration 1024")) ||
        (mismatch = first_mismatch_name(cpu, gpu, &ValidationCheckpoints::final_loop_state, "final memory-loop iteration")) ||
        (mismatch = first_mismatch_name(cpu, gpu, &ValidationCheckpoints::collapsed_text, "post-scratchpad collapse")) ||
        (mismatch = first_mismatch_name(cpu, gpu, &ValidationCheckpoints::post_keccak_state, "full 200-byte post-final Keccak state"))) {
        std::cout << "CryptoNight extended checkpoint FAILED: first divergence at " << mismatch << "\n";
        return;
    }
    std::cout << "CryptoNight extended checkpoints OK through full 200-byte final Keccak state\n";
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
        check(cudaMemset(d_output, 0, sizeof(ValidationCheckpoints)), "cudaMemset CN checkpoint output failed");
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

        if (variant == 0) {
            const auto cpu = core_dark_checkpoints(input, length);
            report_dark_checkpoint_result(cpu, out);
        }
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
