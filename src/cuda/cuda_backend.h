#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace yerbas::cuda {

using Hash512 = std::array<std::uint8_t, 64>;

struct DeviceInfo {
    int id{-1};
    int compute_major{0};
    int compute_minor{0};
    std::size_t total_memory{0};
    int multiprocessors{0};
    int warp_size{0};
};

struct JobDescriptor {
    std::array<std::uint8_t, 80> header{};
    std::array<std::uint8_t, 32> target_le{};
    std::array<std::uint8_t, 18> stages{};
};

struct Candidate {
    std::uint32_t nonce{0};
    std::array<std::uint8_t, 32> hash{};
};

struct StageTiming {
    int stage_index{-1};
    std::uint8_t encoded_stage{0};
    float milliseconds{0.0F};
};

struct BatchProfile {
    std::size_t hashes{0};
    float nonce_init_ms{0.0F};
    float stage_total_ms{0.0F};
    float candidate_ms{0.0F};
    float total_gpu_ms{0.0F};
    std::array<StageTiming, 18> stages{};
};

class BatchEngine {
public:
    explicit BatchEngine(int device_id,
                         std::size_t batch_size = 65536,
                         unsigned int fallback_threads = 1);
    ~BatchEngine();

    BatchEngine(const BatchEngine&) = delete;
    BatchEngine& operator=(const BatchEngine&) = delete;
    BatchEngine(BatchEngine&&) noexcept;
    BatchEngine& operator=(BatchEngine&&) noexcept;

    void upload_job(const JobDescriptor& job);
    std::vector<Candidate> scan(std::uint32_t start_nonce);
    std::vector<Candidate> scan_profiled(std::uint32_t start_nonce, BatchProfile& profile);

    int device_id() const noexcept;
    std::size_t batch_size() const noexcept;
    bool hash_pipeline_ready() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

int device_count();
std::vector<DeviceInfo> enumerate_devices();
void print_devices();

Hash512 blake512_reference_stage(int device_id, const std::uint8_t* input, std::size_t length);
Hash512 bmw512_reference_stage(int device_id, const std::uint8_t* input, std::size_t length);
Hash512 groestl512_reference_stage(int device_id, const std::uint8_t* input, std::size_t length);
Hash512 jh512_reference_stage(int device_id, const std::uint8_t* input, std::size_t length);
Hash512 keccak512_reference_stage(int device_id, const std::uint8_t* input, std::size_t length);
Hash512 skein512_reference_stage(int device_id, const std::uint8_t* input, std::size_t length);
Hash512 luffa512_reference_stage(int device_id, const std::uint8_t* input, std::size_t length);
Hash512 cubehash512_reference_stage(int device_id, const std::uint8_t* input, std::size_t length);
Hash512 shavite512_reference_stage(int device_id, const std::uint8_t* input, std::size_t length);
Hash512 simd512_reference_stage(int device_id, const std::uint8_t* input, std::size_t length);
Hash512 echo512_reference_stage(int device_id, const std::uint8_t* input, std::size_t length);
Hash512 hamsi512_reference_stage(int device_id, const std::uint8_t* input, std::size_t length);
Hash512 fugue512_reference_stage(int device_id, const std::uint8_t* input, std::size_t length);
Hash512 shabal512_reference_stage(int device_id, const std::uint8_t* input, std::size_t length);
Hash512 whirlpool512_reference_stage(int device_id, const std::uint8_t* input, std::size_t length);
Hash512 cryptonight_reference_stage(int device_id,
                                    const std::uint8_t* input,
                                    std::size_t length,
                                    std::uint8_t variant);

} // namespace yerbas::cuda

int yerbas_cuda_device_count();
void yerbas_cuda_print_devices();
