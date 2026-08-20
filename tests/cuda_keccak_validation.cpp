#include "cuda/cuda_backend.h"
#include "cuda/cryptonight/cn_validation.h"
#include "ghostrider/ghostrider.h"
#include "ghostrider_vectors.h"
#include "slow-hash.h"
#include "oaes_lib.h"

extern "C" {
#include "c_keccak.h"
int aesb_single_round(const std::uint8_t* in, std::uint8_t* out, const std::uint8_t* expandedKey);
int aesb_pseudo_round(const std::uint8_t* in, std::uint8_t* out, const std::uint8_t* expandedKey);
}

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
using Hook = yerbas::cuda::Hash512 (*)(int, const std::uint8_t*, std::size_t);

struct ValidationCase {
    std::uint8_t algorithm;
    const char* name;
    Hook hook;
};

constexpr ValidationCase kCases[] = {
    {0, "BLAKE-512",    yerbas::cuda::blake512_reference_stage},
    {1, "BMW-512",      yerbas::cuda::bmw512_reference_stage},
    {2, "Groestl-512",  yerbas::cuda::groestl512_reference_stage},
    {3, "JH-512",       yerbas::cuda::jh512_reference_stage},
    {4, "Keccak-512",   yerbas::cuda::keccak512_reference_stage},
    {5, "Skein-512",    yerbas::cuda::skein512_reference_stage},
    {6, "Luffa-512",    yerbas::cuda::luffa512_reference_stage},
    {7, "CubeHash-512", yerbas::cuda::cubehash512_reference_stage},
    {8, "Shavite-512",  yerbas::cuda::shavite512_reference_stage},
    {9, "SIMD-512",     yerbas::cuda::simd512_reference_stage},
    {10, "Echo-512",    yerbas::cuda::echo512_reference_stage},
    {11, "Hamsi-512",   yerbas::cuda::hamsi512_reference_stage},
    {12, "Fugue-512",   yerbas::cuda::fugue512_reference_stage},
    {13, "Shabal-512",  yerbas::cuda::shabal512_reference_stage},
    {14, "Whirlpool",   yerbas::cuda::whirlpool512_reference_stage},
};

constexpr const char* kCnNames[] = {
    "CN-Dark", "CN-DarkLite", "CN-Fast", "CN-Lite", "CN-Turtle", "CN-TurtleLite"
};

template <typename Container>
std::string hex_string(const Container& value)
{
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (const auto byte : value)
        out << std::setw(2) << static_cast<unsigned int>(byte);
    return out.str();
}

std::uint64_t load64le(const std::uint8_t* p)
{
    return static_cast<std::uint64_t>(p[0]) |
           (static_cast<std::uint64_t>(p[1]) << 8) |
           (static_cast<std::uint64_t>(p[2]) << 16) |
           (static_cast<std::uint64_t>(p[3]) << 24) |
           (static_cast<std::uint64_t>(p[4]) << 32) |
           (static_cast<std::uint64_t>(p[5]) << 40) |
           (static_cast<std::uint64_t>(p[6]) << 48) |
           (static_cast<std::uint64_t>(p[7]) << 56);
}

void store64le(std::uint8_t* p, std::uint64_t v)
{
    for (int i = 0; i < 8; ++i) p[i] = static_cast<std::uint8_t>(v >> (8 * i));
}

void mul128(std::uint64_t a, std::uint64_t b, std::uint64_t& hi, std::uint64_t& lo)
{
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
}

void variant1_mutate(std::uint8_t block[16])
{
    const std::uint8_t tmp = block[11];
    constexpr std::uint32_t table = 0x75310U;
    const std::uint8_t index = static_cast<std::uint8_t>((((tmp >> 3) & 6U) | (tmp & 1U)) << 1);
    block[11] = static_cast<std::uint8_t>(tmp ^ ((table >> index) & 0x30U));
}

yerbas::cuda::cryptonight::ValidationCheckpoints cpu_dark_checkpoints(
    const std::array<std::uint8_t, 64>& input)
{
    constexpr std::size_t page_size = 524288U;
    constexpr std::size_t aes_rounds = 32768U;

    std::array<std::uint8_t, 200> state{};
    keccak1600(input.data(), static_cast<int>(input.size()), state.data());

    OAES_CTX* raw_ctx = oaes_alloc();
    if (raw_ctx == nullptr) throw std::runtime_error("oaes_alloc failed in checkpoint reference");
    auto* ctx = reinterpret_cast<oaes_ctx*>(raw_ctx);
    if (oaes_key_import_data(raw_ctx, state.data(), 32) != OAES_RET_SUCCESS) {
        oaes_free(&raw_ctx);
        throw std::runtime_error("oaes_key_import_data failed in checkpoint reference");
    }

    yerbas::cuda::cryptonight::ValidationCheckpoints out{};
    std::memcpy(out.expanded_key_prefix.data(), ctx->key->exp_data, out.expanded_key_prefix.size());

    std::array<std::uint8_t, 128> text{};
    std::memcpy(text.data(), state.data() + 64, text.size());
    std::vector<std::uint8_t> scratchpad(page_size);
    const std::size_t init_rounds = page_size / text.size();
    for (std::size_t i = 0; i < init_rounds; ++i) {
        for (int block = 0; block < 8; ++block) {
            auto* x = text.data() + block * 16;
            aesb_pseudo_round(x, x, ctx->key->exp_data);
        }
        std::memcpy(scratchpad.data() + i * text.size(), text.data(), text.size());
    }
    std::memcpy(out.scratchpad_prefix.data(), scratchpad.data(), out.scratchpad_prefix.size());

    std::uint8_t a[16], b[16], c[16], t[16];
    for (int i = 0; i < 16; ++i) {
        a[i] = static_cast<std::uint8_t>(state[i] ^ state[32 + i]);
        b[i] = static_cast<std::uint8_t>(state[16 + i] ^ state[48 + i]);
    }
    const std::uint64_t tweak = load64le(input.data() + 35) ^ load64le(state.data() + 192);

    std::size_t j = static_cast<std::size_t>((load64le(a) >> 4) & (aes_rounds - 1));
    std::uint8_t* slot = scratchpad.data() + j * 16U;
    aesb_single_round(slot, c, a);
    for (int k = 0; k < 16; ++k) slot[k] = static_cast<std::uint8_t>(c[k] ^ b[k]);
    variant1_mutate(slot);

    j = static_cast<std::size_t>((load64le(c) >> 4) & (aes_rounds - 1));
    slot = scratchpad.data() + j * 16U;
    std::memcpy(t, slot, 16);

    std::uint64_t hi = 0, lo = 0;
    mul128(load64le(c), load64le(t), hi, lo);
    std::uint64_t a0 = load64le(a) + hi;
    std::uint64_t a1 = load64le(a + 8) + lo;
    store64le(slot, a0);
    store64le(slot + 8, a1 ^ tweak);
    a0 ^= load64le(t);
    a1 ^= load64le(t + 8);
    store64le(a, a0);
    store64le(a + 8, a1);
    std::memcpy(b, c, 16);

    std::memcpy(out.first_loop_state.data(), a, 16);
    std::memcpy(out.first_loop_state.data() + 16, b, 16);
    std::memcpy(out.first_loop_state.data() + 32, c, 16);
    std::memcpy(out.first_loop_state.data() + 48, t, 16);

    oaes_free(&raw_ctx);
    return out;
}
}

int main()
{
    if (yerbas::cuda::device_count() == 0) {
        std::cout << "CUDA core validation skipped: no CUDA device available\n";
        return 0;
    }

    const auto& header = yerbas::test_vectors::MAINNET_GENESIS_HEADER;
    const yerbas::ghostrider::Work header_work{header.data(), header.size()};
    for (const auto& item : kCases) {
        const auto cpu = yerbas::ghostrider::core_hash_reference(header_work, item.algorithm);
        const auto gpu = item.hook(0, header.data(), header.size());
        if (cpu != gpu) {
            std::cerr << "CUDA " << item.name << " mismatch for 80-byte Yerbas header\n";
            return 10 + item.algorithm;
        }
    }

    std::array<std::uint8_t, 64> stage_input{};
    for (std::size_t i = 0; i < stage_input.size(); ++i)
        stage_input[i] = static_cast<std::uint8_t>((i * 37U + 11U) & 0xffU);
    const yerbas::ghostrider::Work stage_work{stage_input.data(), stage_input.size()};

    for (const auto& item : kCases) {
        const auto cpu = yerbas::ghostrider::core_hash_reference(stage_work, item.algorithm);
        const auto gpu = item.hook(0, stage_input.data(), stage_input.size());
        if (cpu != gpu) {
            std::cerr << "CUDA " << item.name << " mismatch for 64-byte intermediate state\n";
            return 30 + item.algorithm;
        }
    }
    std::cout << "CUDA cores 0-14 match pinned Yerbas Core for 80-byte and 64-byte inputs\n";

    std::array<std::uint8_t, 32> cpu_keccak{};
    crypto::cryptonight_dark_fast_hash(
        reinterpret_cast<const char*>(stage_input.data()),
        reinterpret_cast<char*>(cpu_keccak.data()),
        static_cast<std::uint32_t>(stage_input.size()));
    const auto gpu_keccak = yerbas::cuda::cryptonight::validation_keccak_prefix(
        0, stage_input.data(), stage_input.size());

    if (cpu_keccak != gpu_keccak) {
        std::cerr << "CryptoNight checkpoint FAILED: initial Keccak/hash_process state diverges\n"
                  << "  CPU: " << hex_string(cpu_keccak) << "\n"
                  << "  GPU: " << hex_string(gpu_keccak) << "\n";
        return 55;
    }
    std::cout << "CryptoNight checkpoint OK: initial Keccak/hash_process state matches Core\n";

    const auto cpu_cp = cpu_dark_checkpoints(stage_input);
    const auto gpu_cp = yerbas::cuda::cryptonight::validation_checkpoints(
        0, 0, stage_input.data(), stage_input.size());

    if (cpu_cp.expanded_key_prefix != gpu_cp.expanded_key_prefix) {
        std::cerr << "CryptoNight checkpoint FAILED: AES-256 expanded key diverges\n"
                  << "  CPU: " << hex_string(cpu_cp.expanded_key_prefix) << "\n"
                  << "  GPU: " << hex_string(gpu_cp.expanded_key_prefix) << "\n";
        return 56;
    }
    std::cout << "CryptoNight checkpoint OK: AES-256 expanded key matches Core\n";

    if (cpu_cp.scratchpad_prefix != gpu_cp.scratchpad_prefix) {
        std::cerr << "CryptoNight checkpoint FAILED: initial scratchpad fill diverges\n"
                  << "  CPU first128: " << hex_string(cpu_cp.scratchpad_prefix) << "\n"
                  << "  GPU first128: " << hex_string(gpu_cp.scratchpad_prefix) << "\n";
        return 57;
    }
    std::cout << "CryptoNight checkpoint OK: initial scratchpad fill matches Core\n";

    if (cpu_cp.first_loop_state != gpu_cp.first_loop_state) {
        std::cerr << "CryptoNight checkpoint FAILED: first memory-loop iteration diverges\n"
                  << "  CPU a|b|c|t: " << hex_string(cpu_cp.first_loop_state) << "\n"
                  << "  GPU a|b|c|t: " << hex_string(gpu_cp.first_loop_state) << "\n";
        return 58;
    }
    std::cout << "CryptoNight checkpoint OK: first memory-loop iteration matches Core\n";

    std::cout << std::fixed << std::setprecision(3);
    for (std::uint8_t variant = 0; variant < 6; ++variant) {
        std::cout << "Validating " << kCnNames[variant] << "..." << std::flush;
        const std::uint8_t encoded_stage = static_cast<std::uint8_t>(
            yerbas::ghostrider::kCryptoNightStageFlag | variant);

        const auto cpu_start = std::chrono::steady_clock::now();
        const auto cpu = yerbas::ghostrider::stage_reference(stage_work, encoded_stage);
        const auto cpu_stop = std::chrono::steady_clock::now();
        const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_stop - cpu_start).count();

        float kernel_ms = 0.0F;
        const auto gpu_start = std::chrono::steady_clock::now();
        const auto gpu = yerbas::cuda::cryptonight::validation_hash(
            0, variant, stage_input.data(), stage_input.size(), &kernel_ms);
        const auto gpu_stop = std::chrono::steady_clock::now();
        const double gpu_total_ms = std::chrono::duration<double, std::milli>(gpu_stop - gpu_start).count();

        bool match = true;
        for (std::size_t i = 0; i < gpu.size(); ++i) {
            if (gpu[i] != cpu[i]) { match = false; break; }
        }
        if (!match) {
            std::cerr << " FAILED\nCUDA " << kCnNames[variant]
                      << " mismatch for 64-byte intermediate state\n"
                      << "  CPU final: " << hex_string(cpu) << "\n"
                      << "  GPU final: " << hex_string(gpu) << "\n"
                      << "  Early checkpoints matched; divergence occurs after first memory-loop iteration\n";
            return 60 + variant;
        }
        std::cout << " OK | kernel " << kernel_ms << " ms"
                  << " | GPU total " << gpu_total_ms << " ms"
                  << " | CPU " << cpu_ms << " ms\n";
    }

    std::cout << "CUDA CryptoNight variants 0-5 match pinned Yerbas Core\n";
    std::cout << "CryptoNight profiling baseline complete; kernel time excludes allocation/copy overhead.\n";

    std::array<std::uint8_t, 80> full_header = header;
    full_header[76] = 0;
    full_header[77] = 0;
    full_header[78] = 0;
    full_header[79] = 0;
    const yerbas::ghostrider::Work full_work{full_header.data(), full_header.size()};
    const auto cpu_full = yerbas::ghostrider::hash_reference(full_work);

    yerbas::cuda::JobDescriptor job{};
    job.header = full_header;
    job.target_le.fill(0xff);
    job.stages = yerbas::ghostrider::stage_schedule(full_work);

    yerbas::cuda::BatchEngine engine(0, 1);
    engine.upload_job(job);
    const auto candidates = engine.scan(0);
    if (candidates.empty() || candidates.front().nonce != 0) {
        std::cerr << "CUDA full-pipeline validation failed to return nonce 0\n";
        return 80;
    }
    if (candidates.front().hash != cpu_full) {
        std::cerr << "CUDA full GhostRider pipeline DOES NOT match CPU reference for nonce 0\n"
                  << "  CPU: " << hex_string(cpu_full) << "\n"
                  << "  GPU: " << hex_string(candidates.front().hash) << "\n";
        return 81;
    }
    std::cout << "CUDA full 18-stage GhostRider pipeline matches CPU reference for nonce 0\n";
    return 0;
}
