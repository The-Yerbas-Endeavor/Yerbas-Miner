#include "cuda/cuda_backend.h"
#include "cuda/core/stage_dispatch.cuh"
#include "cuda/core_coverage.h"
#include "cuda/cryptonight/cn_config.cuh"
#include "cuda/cryptonight/cn_final.cuh"
#include "cuda/cryptonight/cn_slow_hash.cuh"
#include "cuda/cryptonight/cn_split.cuh"
#include "ghostrider/ghostrider.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace yerbas::cuda {
namespace {

constexpr std::size_t kMaxCandidates = 64;
constexpr std::size_t kStateBytes = 64;
constexpr std::size_t kCnScratchpadStride = cryptonight::max_scratchpad_bytes();
constexpr int kCnGeometryCacheRevision = 4;

struct DeviceJob {
    std::uint8_t header[80];
    std::uint8_t target_le[32];
    std::uint8_t stages[18];
};

struct DeviceCandidate {
    std::uint32_t nonce;
    std::uint8_t hash[32];
};

struct CnGeometry {
    int setup_threads{64};
    int loop_threads{64};
    int final_threads{64};
    int aes_backend{0}; // 0=portable, 1=shared T-table
    float setup_ms{0.0F};
    float loop_ms{0.0F};
    float final_ms{0.0F};
    float portable_aes_ms{0.0F};
    float ttable_aes_ms{0.0F};
};

void check_cuda(cudaError_t status, const char* what)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

std::filesystem::path cn_cache_directory()
{
#ifdef _WIN32
    if (const char* local = std::getenv("LOCALAPPDATA"); local && *local)
        return std::filesystem::path(local) / "Yerbas-Miner";
#endif
    if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg)
        return std::filesystem::path(xdg) / "yerbas-miner";
    if (const char* home = std::getenv("HOME"); home && *home)
        return std::filesystem::path(home) / ".cache" / "yerbas-miner";
    return std::filesystem::current_path() / ".yerbas-miner-cache";
}

std::string sanitize_cache_component(std::string value)
{
    for (char& c : value) {
        const unsigned char u = static_cast<unsigned char>(c);
        if (!((u >= 'a' && u <= 'z') || (u >= 'A' && u <= 'Z') ||
              (u >= '0' && u <= '9') || c == '-' || c == '_')) c = '_';
    }
    return value;
}

std::string cn_cache_key(int device_id,
                         const cudaDeviceProp& props,
                         std::size_t batch_size,
                         int driver_version,
                         int runtime_version)
{
    std::ostringstream out;
    out << "rev" << kCnGeometryCacheRevision
        << "-dev" << device_id
        << '-' << sanitize_cache_component(props.name)
        << "-cc" << props.major << props.minor
        << "-sm" << props.multiProcessorCount
        << "-warp" << props.warpSize
        << "-batch" << batch_size
        << "-drv" << driver_version
        << "-rt" << runtime_version;
    return out.str();
}

bool valid_cn_threads(int value, int max_threads)
{
    return value > 0 && value <= max_threads &&
           (value == 32 || value == 64 || value == 128 || value == 256);
}

void print_cn_phase_measurements(int device_id,
                                 const std::array<CnGeometry, 6>& geometries,
                                 std::size_t sample_count,
                                 const char* source)
{
    std::cout << "[GPU " << device_id << "] CryptoNight phase measurements | source="
              << source << " | sample=" << sample_count << '\n';
    for (std::size_t i = 0; i < geometries.size(); ++i) {
        const auto& g = geometries[i];
        const float total = g.setup_ms + g.loop_ms + g.final_ms;
        const float loop_pct = total > 0.0F ? (100.0F * g.loop_ms / total) : 0.0F;
        std::cout << std::fixed << std::setprecision(3)
                  << "[CUDA CN profile] " << cryptonight::kVariantConfigs[i].name
                  << " | AES=" << (g.aes_backend == 1 ? "ttable" : "portable")
                  << " | geometry=" << g.setup_threads << '/' << g.loop_threads << '/' << g.final_threads
                  << " | setup=" << g.setup_ms << " ms"
                  << " | loop=" << g.loop_ms << " ms"
                  << " | final=" << g.final_ms << " ms"
                  << " | total=" << total << " ms"
                  << " | loop=" << std::setprecision(1) << loop_pct << "%"
                  << std::defaultfloat << '\n';
    }
}

bool load_cn_geometry_cache(int device_id,
                            const cudaDeviceProp& props,
                            std::size_t batch_size,
                            std::size_t sample_count,
                            std::array<CnGeometry, 6>& result)
{
    const char* retune = std::getenv("YERBAS_CUDA_RETUNE");
    if (retune && *retune && std::string(retune) != "0") {
        std::cout << "[GPU " << device_id << "] CryptoNight geometry cache bypassed by YERBAS_CUDA_RETUNE\n";
        return false;
    }

    int driver_version = 0;
    int runtime_version = 0;
    if (cudaDriverGetVersion(&driver_version) != cudaSuccess ||
        cudaRuntimeGetVersion(&runtime_version) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }

    const std::string key = cn_cache_key(device_id, props, batch_size, driver_version, runtime_version);
    const auto path = cn_cache_directory() / ("cn-geometry-" + key + ".txt");
    std::ifstream in(path);
    if (!in) return false;

    std::string magic;
    int revision = 0;
    std::string stored_key;
    std::size_t stored_sample = 0;
    if (!(in >> magic >> revision >> std::quoted(stored_key) >> stored_sample) ||
        magic != "YERBAS_CN_GEOMETRY" || revision != kCnGeometryCacheRevision ||
        stored_key != key || stored_sample != sample_count) {
        return false;
    }

    std::array<CnGeometry, 6> loaded{};
    for (std::size_t i = 0; i < loaded.size(); ++i) {
        std::size_t variant = 0;
        CnGeometry g{};
        if (!(in >> variant >> g.setup_threads >> g.loop_threads >> g.final_threads >> g.aes_backend
                 >> g.setup_ms >> g.loop_ms >> g.final_ms >> g.portable_aes_ms >> g.ttable_aes_ms) ||
            variant != i ||
            !valid_cn_threads(g.setup_threads, props.maxThreadsPerBlock) ||
            !valid_cn_threads(g.loop_threads, props.maxThreadsPerBlock) ||
            !valid_cn_threads(g.final_threads, props.maxThreadsPerBlock) ||
            (g.aes_backend != 0 && g.aes_backend != 1) ||
            g.setup_ms <= 0.0F || g.loop_ms <= 0.0F || g.final_ms <= 0.0F ||
            g.portable_aes_ms <= 0.0F || g.ttable_aes_ms <= 0.0F) {
            return false;
        }
        loaded[i] = g;
    }

    result = loaded;
    std::cout << "[GPU " << device_id << "] CryptoNight geometry cache loaded | " << path.string() << '\n';
    print_cn_phase_measurements(device_id, result, sample_count, "cache");
    return true;
}

void save_cn_geometry_cache(int device_id,
                            const cudaDeviceProp& props,
                            std::size_t batch_size,
                            std::size_t sample_count,
                            const std::array<CnGeometry, 6>& geometries)
{
    int driver_version = 0;
    int runtime_version = 0;
    if (cudaDriverGetVersion(&driver_version) != cudaSuccess ||
        cudaRuntimeGetVersion(&runtime_version) != cudaSuccess) {
        cudaGetLastError();
        return;
    }

    try {
        const auto dir = cn_cache_directory();
        std::filesystem::create_directories(dir);
        const std::string key = cn_cache_key(device_id, props, batch_size, driver_version, runtime_version);
        const auto path = dir / ("cn-geometry-" + key + ".txt");
        const auto temp = path.string() + ".tmp";
        std::ofstream out(temp, std::ios::trunc);
        if (!out) return;
        out << "YERBAS_CN_GEOMETRY " << kCnGeometryCacheRevision << ' '
            << std::quoted(key) << ' ' << sample_count << '\n';
        out << std::setprecision(9);
        for (std::size_t i = 0; i < geometries.size(); ++i) {
            const auto& g = geometries[i];
            out << i << ' ' << g.setup_threads << ' ' << g.loop_threads << ' ' << g.final_threads
                << ' ' << g.aes_backend
                << ' ' << g.setup_ms << ' ' << g.loop_ms << ' ' << g.final_ms
                << ' ' << g.portable_aes_ms << ' ' << g.ttable_aes_ms << '\n';
        }
        out.close();
        if (!out) return;
        std::error_code ec;
        std::filesystem::rename(temp, path, ec);
        if (ec) {
            std::filesystem::remove(path, ec);
            ec.clear();
            std::filesystem::rename(temp, path, ec);
        }
        if (!ec)
            std::cout << "[GPU " << device_id << "] CryptoNight geometry cache saved | " << path.string() << '\n';
    } catch (const std::exception& e) {
        std::cerr << "[GPU " << device_id << "] CryptoNight geometry cache write skipped: " << e.what() << '\n';
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
    if (index < count) nonces[index] = start_nonce + static_cast<std::uint32_t>(index);
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
        ok = core::dispatch_core512(algorithm, states + index * kStateBytes, kStateBytes, digest);
    }
    if (!ok) return;
    std::uint8_t* output = states + index * kStateBytes;
    #pragma unroll
    for (int i = 0; i < 64; ++i) output[i] = digest[i];
}

__global__ void cryptonight_stage(std::uint8_t* states,
                                  std::size_t count,
                                  std::uint8_t variant,
                                  std::uint8_t* scratchpads)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const auto cfg = cryptonight::config_value(variant);
    const std::size_t scratchpad_stride = static_cast<std::size_t>(cfg.page_size);
    std::uint8_t* state = states + index * kStateBytes;
    std::uint8_t* scratchpad = scratchpads + index * scratchpad_stride;
    std::uint8_t digest[32];
    if (!cryptonight::slow_hash(variant, state, kStateBytes, scratchpad, digest)) return;
    #pragma unroll
    for (int i = 0; i < 32; ++i) state[i] = digest[i];
    #pragma unroll
    for (int i = 32; i < 64; ++i) state[i] = 0;
}

template <std::uint8_t VariantIndex, bool UseTTable = false>
__global__ void cryptonight_setup_stage(std::uint8_t* states,
                                        std::size_t count,
                                        std::uint8_t* scratchpads,
                                        cryptonight::SplitContext* contexts)
{
    __shared__ std::uint32_t aes_tables[1024];
    if constexpr (UseTTable) {
        cryptonight::init_shared_aes_ttables(aes_tables);
        __syncthreads();
    }
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    std::uint8_t* state = states + index * kStateBytes;
    std::uint8_t* scratchpad = scratchpads + index * static_cast<std::size_t>(cfg.page_size);
    cryptonight::split_setup<VariantIndex, UseTTable>(state, kStateBytes, scratchpad, contexts[index],
                                                      UseTTable ? aes_tables : nullptr);
}

template <std::uint8_t VariantIndex>
__global__ void cryptonight_loop_stage(std::size_t count,
                                       std::uint8_t* scratchpads,
                                       const cryptonight::SplitContext* contexts)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    std::uint8_t* scratchpad = scratchpads + index * static_cast<std::size_t>(cfg.page_size);
    cryptonight::split_memory_loop<VariantIndex>(scratchpad, contexts[index]);
}

template <std::uint8_t VariantIndex, bool UseTTable = false>
__global__ void cryptonight_final_stage(std::uint8_t* states,
                                        std::size_t count,
                                        std::uint8_t* scratchpads,
                                        cryptonight::SplitContext* contexts)
{
    __shared__ std::uint32_t aes_tables[1024];
    if constexpr (UseTTable) {
        cryptonight::init_shared_aes_ttables(aes_tables);
        __syncthreads();
    }
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    constexpr auto cfg = cryptonight::config_value(VariantIndex);
    std::uint8_t* state = states + index * kStateBytes;
    std::uint8_t* scratchpad = scratchpads + index * static_cast<std::size_t>(cfg.page_size);
    std::uint8_t digest[32];
    cryptonight::split_finalize<VariantIndex, UseTTable>(scratchpad, contexts[index], digest,
                                                         UseTTable ? aes_tables : nullptr);
    #pragma unroll
    for (int i = 0; i < 32; ++i) state[i] = digest[i];
    #pragma unroll
    for (int i = 32; i < 64; ++i) state[i] = 0;
}

template <typename Launch>
float time_cuda_phase(cudaStream_t stream, Launch&& launch)
{
    cudaEvent_t start{}, stop{};
    check_cuda(cudaEventCreate(&start), "cudaEventCreate autotune start failed");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate autotune stop failed");
    float milliseconds = 0.0F;
    try {
        check_cuda(cudaEventRecord(start, stream), "cudaEventRecord autotune start failed");
        launch();
        check_cuda(cudaEventRecord(stop, stream), "cudaEventRecord autotune stop failed");
        check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize autotune failed");
        check_cuda(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime autotune failed");
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        return milliseconds;
    } catch (...) {
        cudaEventDestroy(stop);
        cudaEventDestroy(start);
        throw;
    }
}

