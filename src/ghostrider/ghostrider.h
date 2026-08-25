#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace yerbas::ghostrider {

using Hash256 = std::array<std::uint8_t, 32>;
using Hash512 = std::array<std::uint8_t, 64>;
using StageSchedule = std::array<std::uint8_t, 18>;

constexpr std::uint8_t kCryptoNightStageFlag = 0x80;

struct Work {
    const std::uint8_t* data{nullptr};
    std::size_t size{0};
};

Hash256 hash_reference(const Work& work);
Hash256 hash_optimized(const Work& work);
bool optimized_cpu_ready() noexcept;

// Production multilaned CPU path. All works must belong to the same Stratum
// job/schedule. count must be 1..4. Per-CN widths are the qualified 1/2/4-way
// choices for Dark, DarkLite, Fast, Lite, Turtle and TurtleLite.
bool hash_optimized_batch(const Work* works,
                          Hash256* outputs,
                          std::size_t count,
                          const std::array<unsigned int, 6>& cn_widths);

// Stage-level primitives used by the cooperative CPU scheduler. Core stages
// remain fully independent across workers; only CryptoNight stages rendezvous
// into 2-way pairs. These APIs preserve the exact GhostRider stage semantics
// and are intended to be parity/performance gated before production use.
bool optimized_core_stage(const Work& input,
                          std::uint8_t algorithm,
                          Hash512& output) noexcept;
bool optimized_cn_stage(const Hash512& input,
                        std::uint8_t variant,
                        Hash512& output) noexcept;
bool optimized_cn_pair_stage(const Hash512& input0,
                             const Hash512& input1,
                             std::uint8_t variant,
                             Hash512& output0,
                             Hash512& output1) noexcept;

Hash256 hash_staged_reference(const Work& work);
Hash512 core_hash_reference(const Work& work, int algorithm);
Hash512 stage_reference(const Work& work, std::uint8_t stage);
StageSchedule stage_schedule(const Work& work);
StageSchedule stage_schedule_quiet(const Work& work);
std::uint64_t schedule_fingerprint(const StageSchedule& schedule) noexcept;
const char* cryptonight_name(std::uint8_t index) noexcept;
bool reference_ready() noexcept;

} // namespace yerbas::ghostrider
