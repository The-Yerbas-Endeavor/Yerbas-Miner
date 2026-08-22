#include "cuda/cryptonight/cn_config.cuh"
#include "cuda/cryptonight/cn_final.cuh"
#include "cuda/cryptonight/cn_slow_hash.cuh"
#include "cuda/cryptonight/cn_split.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cn = yerbas::cuda::cryptonight;

namespace {

constexpr std::uint8_t kVariant = 2; // CN-Fast
constexpr std::size_t kStateBytes = 64;

void check(cudaError_t status, const char* what)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

template <std::uint8_t VariantIndex>
__global__ void setup_kernel(const std::uint8_t* inputs,
                             std::size_t count,
                             std::uint8_t* scratchpads,
                             cn::SplitContext* contexts)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    constexpr auto cfg = cn::config_value(VariantIndex);
    cn::split_setup<VariantIndex>(inputs + index * kStateBytes,
                                  kStateBytes,
                                  scratchpads + index * static_cast<std::size_t>(cfg.page_size),
                                  contexts[index]);
}

template <std::uint8_t VariantIndex>
__global__ void loop_kernel(std::size_t count,
                            std::uint8_t* scratchpads,
                            const cn::SplitContext* contexts)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    constexpr auto cfg = cn::config_value(VariantIndex);
    cn::split_memory_loop<VariantIndex>(
        scratchpads + index * static_cast<std::size_t>(cfg.page_size), contexts[index]);
}

template <std::uint8_t VariantIndex>
__global__ void final_kernel(std::size_t count,
                             std::uint8_t* scratchpads,
                             cn::SplitContext* contexts,
                             std::uint8_t* outputs)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    constexpr auto cfg = cn::config_value(VariantIndex);
    cn::split_finalize<VariantIndex>(
        scratchpads + index * static_cast<std::size_t>(cfg.page_size),
        contexts[index], outputs + index * 32U);
}

__global__ void monolithic_one(const std::uint8_t* input,
                               std::uint8_t* scratchpad,
                               std::uint8_t* output)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    cn::slow_hash_specialized<kVariant>(input, kStateBytes, scratchpad, output);
}

float elapsed(cudaEvent_t start, cudaEvent_t stop)
{
    float ms = 0.0F;
    check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime failed");
    return ms;
}

struct Timings {
    int setup_threads{0};
    int loop_threads{0};
    int final_threads{0};
    float setup_ms{0.0F};
    float loop_ms{0.0F};
    float final_ms{0.0F};
    float total_ms{0.0F};
};

Timings run_profile(std::size_t count,
                    int setup_threads,
                    int loop_threads,
                    int final_threads,
                    const std::uint8_t* d_inputs,
                    std::uint8_t* d_scratch,
                    cn::SplitContext* d_contexts,
                    std::uint8_t* d_outputs)
{
    const int setup_blocks = static_cast<int>((count + static_cast<std::size_t>(setup_threads) - 1U) /
                                              static_cast<std::size_t>(setup_threads));
    const int loop_blocks = static_cast<int>((count + static_cast<std::size_t>(loop_threads) - 1U) /
                                             static_cast<std::size_t>(loop_threads));
    const int final_blocks = static_cast<int>((count + static_cast<std::size_t>(final_threads) - 1U) /
                                              static_cast<std::size_t>(final_threads));

    cudaEvent_t e0{}, e1{}, e2{}, e3{};
    check(cudaEventCreate(&e0), "cudaEventCreate e0 failed");
    check(cudaEventCreate(&e1), "cudaEventCreate e1 failed");
    check(cudaEventCreate(&e2), "cudaEventCreate e2 failed");
    check(cudaEventCreate(&e3), "cudaEventCreate e3 failed");

    setup_kernel<kVariant><<<setup_blocks, setup_threads>>>(d_inputs, count, d_scratch, d_contexts);
    loop_kernel<kVariant><<<loop_blocks, loop_threads>>>(count, d_scratch, d_contexts);
    final_kernel<kVariant><<<final_blocks, final_threads>>>(count, d_scratch, d_contexts, d_outputs);
    check(cudaDeviceSynchronize(), "warmup failed");

    check(cudaEventRecord(e0), "record setup start failed");
    setup_kernel<kVariant><<<setup_blocks, setup_threads>>>(d_inputs, count, d_scratch, d_contexts);
    check(cudaGetLastError(), "setup launch failed");
    check(cudaEventRecord(e1), "record setup stop failed");
    loop_kernel<kVariant><<<loop_blocks, loop_threads>>>(count, d_scratch, d_contexts);
    check(cudaGetLastError(), "loop launch failed");
    check(cudaEventRecord(e2), "record loop stop failed");
    final_kernel<kVariant><<<final_blocks, final_threads>>>(count, d_scratch, d_contexts, d_outputs);
    check(cudaGetLastError(), "final launch failed");
    check(cudaEventRecord(e3), "record final stop failed");
    check(cudaEventSynchronize(e3), "profile synchronize failed");

    Timings out{};
    out.setup_threads = setup_threads;
    out.loop_threads = loop_threads;
    out.final_threads = final_threads;
    out.setup_ms = elapsed(e0, e1);
    out.loop_ms = elapsed(e1, e2);
    out.final_ms = elapsed(e2, e3);
    out.total_ms = out.setup_ms + out.loop_ms + out.final_ms;

    cudaEventDestroy(e3); cudaEventDestroy(e2); cudaEventDestroy(e1); cudaEventDestroy(e0);
    return out;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const int device = argc > 1 ? std::atoi(argv[1]) : 0;
        const std::size_t count = argc > 2 ? static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10)) : 3584U;
        if (count == 0) throw std::runtime_error("count must be positive");

        std::vector<int> thread_values;
        for (int i = 3; i < argc; ++i) {
            const int value = std::atoi(argv[i]);
            if (value > 0) thread_values.push_back(value);
        }
        if (thread_values.empty()) thread_values = {32, 64, 96, 128, 192, 256};

        check(cudaSetDevice(device), "cudaSetDevice failed");
        cudaDeviceProp props{};
        check(cudaGetDeviceProperties(&props, device), "cudaGetDeviceProperties failed");

        constexpr auto cfg = cn::config_value(kVariant);
        const std::size_t scratch_bytes = count * static_cast<std::size_t>(cfg.page_size);

        std::vector<std::uint8_t> host_inputs(count * kStateBytes);
        for (std::size_t i = 0; i < count; ++i) {
            for (std::size_t b = 0; b < kStateBytes; ++b)
                host_inputs[i * kStateBytes + b] = static_cast<std::uint8_t>((b * 17U + i * 29U + 0x5aU) & 0xffU);
        }

        std::uint8_t* d_inputs = nullptr;
        std::uint8_t* d_scratch = nullptr;
        std::uint8_t* d_outputs = nullptr;
        cn::SplitContext* d_contexts = nullptr;
        std::uint8_t* d_reference_scratch = nullptr;
        std::uint8_t* d_reference_output = nullptr;

        check(cudaMalloc(reinterpret_cast<void**>(&d_inputs), host_inputs.size()), "cudaMalloc inputs failed");
        check(cudaMalloc(reinterpret_cast<void**>(&d_scratch), scratch_bytes), "cudaMalloc scratchpads failed");
        check(cudaMalloc(reinterpret_cast<void**>(&d_outputs), count * 32U), "cudaMalloc outputs failed");
        check(cudaMalloc(reinterpret_cast<void**>(&d_contexts), count * sizeof(cn::SplitContext)), "cudaMalloc contexts failed");
        check(cudaMalloc(reinterpret_cast<void**>(&d_reference_scratch), cfg.page_size), "cudaMalloc reference scratchpad failed");
        check(cudaMalloc(reinterpret_cast<void**>(&d_reference_output), 32U), "cudaMalloc reference output failed");
        check(cudaMemcpy(d_inputs, host_inputs.data(), host_inputs.size(), cudaMemcpyHostToDevice), "cudaMemcpy inputs failed");

        std::cout << "Yerbas CUDA CN-Fast phase thread sweep\n"
                  << "GPU: " << device << " | " << props.name << " | CC " << props.major << '.' << props.minor << "\n"
                  << "Batch: " << count << " | Scratchpad: " << std::fixed << std::setprecision(2)
                  << (static_cast<double>(scratch_bytes) / (1024.0 * 1024.0 * 1024.0)) << " GiB\n\n";

        Timings best{};
        best.total_ms = 1.0e30F;
        for (const int threads : thread_values) {
            if (threads > props.maxThreadsPerBlock) continue;
            const Timings t = run_profile(count, threads, threads, threads,
                                          d_inputs, d_scratch, d_contexts, d_outputs);
            const double hps = static_cast<double>(count) * 1000.0 / t.total_ms;
            std::cout << "threads " << std::setw(3) << threads
                      << " | setup " << std::setprecision(3) << t.setup_ms
                      << " ms | loop " << t.loop_ms
                      << " ms | final " << t.final_ms
                      << " ms | total " << t.total_ms
                      << " ms | " << std::setprecision(2) << hps << " H/s\n";
            if (t.total_ms < best.total_ms) best = t;
        }

        std::cout << "\nBest same-thread geometry: " << best.loop_threads
                  << " threads | " << std::setprecision(3) << best.total_ms << " ms | "
                  << std::setprecision(2) << (static_cast<double>(count) * 1000.0 / best.total_ms)
                  << " CN-Fast H/s\n";

        // Verify split output for hash 0 against the still-present monolithic implementation.
        monolithic_one<<<1, 1>>>(d_inputs, d_reference_scratch, d_reference_output);
        check(cudaDeviceSynchronize(), "monolithic reference failed");
        std::array<std::uint8_t, 32> split_out{}, reference_out{};
        check(cudaMemcpy(split_out.data(), d_outputs, 32U, cudaMemcpyDeviceToHost), "copy split output failed");
        check(cudaMemcpy(reference_out.data(), d_reference_output, 32U, cudaMemcpyDeviceToHost), "copy reference output failed");
        const bool matches = split_out == reference_out;
        std::cout << "split matches monolithic hash 0: " << (matches ? "YES" : "NO") << "\n";

        cudaFree(d_reference_output); cudaFree(d_reference_scratch); cudaFree(d_contexts);
        cudaFree(d_outputs); cudaFree(d_scratch); cudaFree(d_inputs);
        return matches ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "CN-Fast phase profiler failed: " << e.what() << "\n";
        return 1;
    }
}
