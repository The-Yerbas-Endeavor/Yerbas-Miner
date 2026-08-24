#include "ghostrider/ghostrider.h"

#include <array>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "hash_selection.h"
#include "uint256.h"

namespace yerbas::ghostrider {

namespace {
constexpr std::size_t kVersionSize = 4;
constexpr std::size_t kPrevHashSize = 32;
constexpr std::size_t kMinimumHeaderPrefix = kVersionSize + kPrevHashSize;

uint256 previous_block_hash(const Work& work)
{
    if (work.data == nullptr || work.size < kMinimumHeaderPrefix) {
        throw std::invalid_argument(
            "GhostRider input must contain at least nVersion and hashPrevBlock");
    }

    uint256 prev_block_hash;
    std::memcpy(prev_block_hash.begin(), work.data + kVersionSize, kPrevHashSize);
    return prev_block_hash;
}

void selections(const Work& work,
                std::vector<int>& core_hash_indexes,
                std::vector<int>& random_cns)
{
    HashSelection hash_selection(
        previous_block_hash(work),
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
        {0, 1, 2, 3, 4, 5});
    random_cns = hash_selection.getCnIndexes();
    core_hash_indexes = hash_selection.getAlgoIndexes();
}

StageSchedule compute_schedule(const Work& work)
{
    std::vector<int> core_hash_indexes;
    std::vector<int> random_cns;
    selections(work, core_hash_indexes, random_cns);

    StageSchedule schedule{};
    for (int i = 0; i < 18; ++i) {
        if (i < 5) schedule[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(core_hash_indexes[i]);
        else if (i == 5) schedule[5] = static_cast<std::uint8_t>(kCryptoNightStageFlag | random_cns[0]);
        else if (i < 11) schedule[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(core_hash_indexes[i - 1]);
        else if (i == 11) schedule[11] = static_cast<std::uint8_t>(kCryptoNightStageFlag | random_cns[1]);
        else if (i < 17) schedule[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(core_hash_indexes[i - 2]);
        else schedule[17] = static_cast<std::uint8_t>(kCryptoNightStageFlag | random_cns[2]);
    }
    return schedule;
}

const StageSchedule& cached_schedule(const Work& work)
{
    if (work.data == nullptr || work.size < kMinimumHeaderPrefix)
        throw std::invalid_argument("GhostRider work buffer is too short");

    struct Cache {
        std::array<std::uint8_t, kPrevHashSize> prevhash{};
        StageSchedule schedule{};
        bool valid{false};
    };
    thread_local Cache cache;

    const auto* prev = work.data + kVersionSize;
    if (!cache.valid || std::memcmp(cache.prevhash.data(), prev, kPrevHashSize) != 0) {
        std::memcpy(cache.prevhash.data(), prev, kPrevHashSize);
        cache.schedule = compute_schedule(work);
        cache.valid = true;
    }
    return cache.schedule;
}

const char* core_name(std::uint8_t index)
{
    static constexpr const char* names[15] = {
        "BLAKE", "BMW", "Groestl", "JH", "Keccak",
        "Skein", "Luffa", "CubeHash", "Shavite", "SIMD",
        "Echo", "Hamsi", "Fugue", "Shabal", "Whirlpool"
    };
    return index < 15 ? names[index] : "Core?";
}

const char* cn_name(std::uint8_t index)
{
    static constexpr const char* names[6] = {
        "CN-Dark", "CN-DarkLite", "CN-Fast",
        "CN-Lite", "CN-Turtle", "CN-TurtleLite"
    };
    return index < 6 ? names[index] : "CN?";
}

std::uint64_t schedule_fingerprint64(const StageSchedule& schedule)
{
    std::uint64_t value = 14695981039346656037ULL;
    for (const auto stage : schedule) {
        value ^= static_cast<std::uint64_t>(stage);
        value *= 1099511628211ULL;
    }
    return value;
}

void log_schedule(const StageSchedule& schedule)
{
    std::ostringstream compact;
    compact << std::hex << std::setfill('0');
    for (std::size_t i = 0; i < schedule.size(); ++i) {
        if (i) compact << '-';
        compact << std::setw(2) << static_cast<unsigned int>(schedule[i]);
    }

    std::ostringstream names;
    for (std::size_t i = 0; i < schedule.size(); ++i) {
        if (i) names << ' ';
        const std::uint8_t stage = schedule[i];
        if ((stage & kCryptoNightStageFlag) != 0)
            names << i << ':' << cn_name(static_cast<std::uint8_t>(stage & 0x7fU));
        else
            names << i << ':' << core_name(stage);
    }

    std::ostringstream fingerprint;
    fingerprint << std::hex << std::setfill('0') << std::setw(16)
                << schedule_fingerprint64(schedule);

    std::cout << "[GhostRider] schedule fingerprint=" << fingerprint.str()
              << " | stages=" << compact.str() << '\n'
              << "[GhostRider] schedule " << names.str() << '\n';
}
} // namespace

Hash512 core_hash_reference(const Work& work, int algorithm)
{
    if (work.data == nullptr || work.size == 0) throw std::invalid_argument("Core hash input must not be empty");
    if (work.size > static_cast<std::size_t>(std::numeric_limits<int>::max())) throw std::invalid_argument("Core hash work buffer is too large");
    if (algorithm < 0 || algorithm > 14) throw std::invalid_argument("Core hash algorithm index must be 0..14");

    uint512 result;
    coreHash(work.data, &result, static_cast<int>(work.size), algorithm);
    Hash512 out{};
    std::memcpy(out.data(), result.begin(), out.size());
    return out;
}

Hash512 stage_reference(const Work& work, std::uint8_t stage)
{
    if (work.data == nullptr || work.size == 0) throw std::invalid_argument("GhostRider stage input must not be empty");
    if (work.size > static_cast<std::size_t>(std::numeric_limits<int>::max())) throw std::invalid_argument("GhostRider stage work buffer is too large");

    uint512 result;
    if ((stage & kCryptoNightStageFlag) != 0) {
        const int variant = static_cast<int>(stage & 0x7fU);
        if (variant < 0 || variant > 5) throw std::invalid_argument("CryptoNight stage index must be 0..5");
        if (work.size != 64) throw std::invalid_argument("CryptoNight GhostRider stage input must be 64 bytes");
        uint512 input;
        std::memcpy(input.begin(), work.data, 64);
        cnHash(&input, &result, 64, variant);
    } else {
        const int algorithm = static_cast<int>(stage);
        if (algorithm < 0 || algorithm > 14) throw std::invalid_argument("Core GhostRider stage index must be 0..14");
        coreHash(work.data, &result, static_cast<int>(work.size), algorithm);
    }

    Hash512 out{};
    std::memcpy(out.data(), result.begin(), out.size());
    return out;
}

StageSchedule stage_schedule(const Work& work)
{
    const auto schedule = compute_schedule(work);
    log_schedule(schedule);
    return schedule;
}

StageSchedule stage_schedule_quiet(const Work& work)
{
    return compute_schedule(work);
}

std::uint64_t schedule_fingerprint(const StageSchedule& schedule) noexcept
{
    return schedule_fingerprint64(schedule);
}

const char* cryptonight_name(std::uint8_t index) noexcept
{
    return cn_name(index);
}

Hash256 hash_staged_reference(const Work& work)
{
    if (work.data == nullptr || work.size == 0) throw std::invalid_argument("GhostRider work buffer must not be empty");

    const auto schedule = compute_schedule(work);
    Hash512 state{};
    for (std::size_t i = 0; i < schedule.size(); ++i) {
        if (i == 0) {
            state = stage_reference(work, schedule[i]);
        } else {
            const Work stage_work{state.data(), state.size()};
            state = stage_reference(stage_work, schedule[i]);
        }
    }

    Hash256 out{};
    std::memcpy(out.data(), state.data(), out.size());
    return out;
}

Hash256 hash_reference(const Work& work)
{
    if (work.data == nullptr || work.size == 0)
        throw std::invalid_argument("GhostRider work buffer must not be empty");
    if (work.size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("GhostRider work buffer is too large");

    // The GhostRider schedule depends only on hashPrevBlock (header bytes 4..35),
    // not on the nonce. CPU miners hash thousands of nonces for the same job, so
    // recomputing HashSelection and allocating its vectors for every nonce wastes
    // significant host time. Cache the 18-stage schedule independently in each
    // worker thread and refresh it only when the pool changes previous block hash.
    const auto& schedule = cached_schedule(work);

    uint512 hash[18];
    for (std::size_t i = 0; i < schedule.size(); ++i) {
        const std::uint8_t stage = schedule[i];
        const bool is_cn = (stage & kCryptoNightStageFlag) != 0;
        const int algorithm = static_cast<int>(stage & 0x7fU);

        if (is_cn) {
            // GhostRider's CN positions are 5, 11 and 17, so the input is always
            // the preceding 64-byte state.
            cnHash(&hash[i - 1], &hash[i], 64, algorithm);
        } else {
            const void* input = (i == 0)
                ? static_cast<const void*>(work.data)
                : static_cast<const void*>(&hash[i - 1]);
            const int length = (i == 0) ? static_cast<int>(work.size) : 64;
            coreHash(input, &hash[i], length, algorithm);
        }
    }

    const uint256 result = hash[17].trim256();
    Hash256 out{};
    std::memcpy(out.data(), result.begin(), out.size());
    return out;
}

bool reference_ready() noexcept
{
    return true;
}

} // namespace yerbas::ghostrider
