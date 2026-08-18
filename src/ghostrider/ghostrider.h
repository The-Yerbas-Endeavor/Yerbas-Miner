#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace yerbas::ghostrider {

using Hash256 = std::array<std::uint8_t, 32>;
using Hash512 = std::array<std::uint8_t, 64>;
using StageSchedule = std::array<std::uint8_t, 18>;

// Stage encoding used by the GPU scheduler:
//   0x00..0x0e = one of the 15 core hashes
//   0x80..0x85 = one of the 6 CryptoNight variants
constexpr std::uint8_t kCryptoNightStageFlag = 0x80;

struct Work {
    const std::uint8_t* data{nullptr};
    std::size_t size{0};
};

// CPU reference implementation wired to the exact GhostRider primitives and
// hash-selection logic from Yerbas Core. For normal mining input, data is the
// serialized 80-byte Yerbas block header (nVersion through nNonce).
Hash256 hash_reference(const Work& work);

// Exposes one of the 15 512-bit core hashes for GPU validation. algorithm must
// be in the same 0..14 index space used by Yerbas Core's coreHash().
Hash512 core_hash_reference(const Work& work, int algorithm);

// Run exactly one encoded GhostRider stage using the pinned Yerbas Core
// implementation. This is used by the CUDA bootstrap pipeline as a correctness
// fallback until every selectable core/CryptoNight stage has a native kernel.
// Core stages use 0x00..0x0e; CryptoNight stages use 0x80..0x85.
Hash512 stage_reference(const Work& work, std::uint8_t stage);

// GhostRider selection depends only on hashPrevBlock, so one schedule can be
// computed once per Stratum job and reused for every nonce in the GPU batch.
StageSchedule stage_schedule(const Work& work);

bool reference_ready() noexcept;

} // namespace yerbas::ghostrider
