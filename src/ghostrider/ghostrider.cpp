#include "ghostrider/ghostrider.h"
#include "cpu/cn_2way.h"

#include <array>
#include <atomic>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "hash_selection.h"
#include "uint256.h"

extern "C" void yerbas_cn_slow_hash_reuse(const char* input,
                                           char* output,
                                           uint32_t len,
                                           int variant,
                                           uint32_t page_size,
                                           uint32_t iterations,
                                           size_t aes_rounds);

namespace yerbas::ghostrider {

namespace {
constexpr std::size_t kVersionSize = 4;
constexpr std::size_t kPrevHashSize = 32;
constexpr std::size_t kMinimumHeaderPrefix = kVersionSize + kPrevHashSize;

struct CnParams {
    std::uint32_t page_size;
    std::uint32_t iterations;
    std::size_t aes_rounds;
};

constexpr std::array<CnParams, 6> kCnParams{{
    {524288U, 131072U, 32768U},
    {524288U, 131072U, 16384U},
    {2097152U, 262144U, 131072U},
    {1048576U, 262144U, 65536U},
    {262144U, 65536U, 16384U},
    {262144U, 65536U, 8192U},
}};

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

bool reusable_cn_hash(const uint512& input, uint512& output, int variant) noexcept
{
    if (variant < 0 || variant >= static_cast<int>(kCnParams.size())) return false;
    output = uint512{};
    const auto& params = kCnParams[static_cast<std::size_t>(variant)];
    yerbas_cn_slow_hash_reuse(reinterpret_cast<const char*>(&input),
                              reinterpret_cast<char*>(&output),
                              64U,
                              1,
                              params.page_size,
                              params.iterations,
                              params.aes_rounds);
    return true;
}

bool validate_reusable_cn() noexcept
{
    uint512 input{};
    for (std::size_t i = 0; i < 64; ++i)
        input.begin()[i] = static_cast<unsigned char>((i * 37U + 11U) & 0xffU);

    for (int variant = 0; variant < 6; ++variant) {
        uint512 reference{};
        uint512 candidate{};
        cnHash(&input, &reference, 64, variant);
        if (!reusable_cn_hash(input, candidate, variant)) return false;
        if (std::memcmp(reference.begin(), candidate.begin(), 64) != 0)
            return false;
    }
    return true;
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

    const auto& schedule = cached_schedule(work);

    uint512 hash[18];
    for (std::size_t i = 0; i < schedule.size(); ++i) {
        const std::uint8_t stage = schedule[i];
        const bool is_cn = (stage & kCryptoNightStageFlag) != 0;
        const int algorithm = static_cast<int>(stage & 0x7fU);

        if (is_cn) {
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

bool optimized_cpu_ready() noexcept
{
    static std::atomic<int> state{0};
    int current = state.load(std::memory_order_acquire);
    if (current == 2) return true;
    if (current == 3) return false;

    int expected = 0;
    if (state.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
        const bool ready = validate_reusable_cn();
        std::cout << "CPU CryptoNight reusable contexts: "
                  << (ready ? "parity PASS (6/6)" : "parity FAIL; reference fallback")
                  << '\n';
        state.store(ready ? 2 : 3, std::memory_order_release);
        return ready;
    }

    do {
        current = state.load(std::memory_order_acquire);
    } while (current == 1);
    return current == 2;
}

Hash256 hash_optimized(const Work& work)
{
    if (!optimized_cpu_ready()) return hash_reference(work);
    if (work.data == nullptr || work.size == 0) return hash_reference(work);
    if (work.size > static_cast<std::size_t>(std::numeric_limits<int>::max())) return hash_reference(work);

    const auto& schedule = cached_schedule(work);
    uint512 hash[18];
    for (std::size_t i = 0; i < schedule.size(); ++i) {
        const std::uint8_t stage = schedule[i];
        const bool is_cn = (stage & kCryptoNightStageFlag) != 0;
        const int algorithm = static_cast<int>(stage & 0x7fU);
        if (is_cn) {
            if (!reusable_cn_hash(hash[i - 1], hash[i], algorithm)) return hash_reference(work);
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

bool hash_optimized_batch(const Work* works,
                          Hash256* outputs,
                          std::size_t count,
                          const std::array<unsigned int, 6>& cn_widths)
{
    if (works == nullptr || outputs == nullptr || count == 0 || count > 4) return false;
    if (!optimized_cpu_ready()) return false;
    for (std::size_t lane = 0; lane < count; ++lane) {
        if (works[lane].data == nullptr || works[lane].size == 0 ||
            works[lane].size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            return false;
    }

    const auto& schedule = cached_schedule(works[0]);
    uint512 hash[4][18]{};

    for (std::size_t i = 0; i < schedule.size(); ++i) {
        const std::uint8_t stage = schedule[i];
        const bool is_cn = (stage & kCryptoNightStageFlag) != 0;
        const int algorithm = static_cast<int>(stage & 0x7fU);

        if (!is_cn) {
            for (std::size_t lane = 0; lane < count; ++lane) {
                const void* input = (i == 0)
                    ? static_cast<const void*>(works[lane].data)
                    : static_cast<const void*>(&hash[lane][i - 1]);
                const int length = (i == 0) ? static_cast<int>(works[lane].size) : 64;
                coreHash(input, &hash[lane][i], length, algorithm);
            }
            continue;
        }

        if (algorithm < 0 || algorithm >= static_cast<int>(kCnParams.size())) return false;
        const auto& params = kCnParams[static_cast<std::size_t>(algorithm)];
        const unsigned int width = cn_widths[static_cast<std::size_t>(algorithm)];
        std::size_t lane = 0;

        if (width >= 4U && count == 4U) {
            for (std::size_t l = 0; l < 4; ++l) hash[l][i] = uint512{};
            if (!yerbas_cn_hash_quad_4way(
                    reinterpret_cast<const char*>(&hash[0][i - 1]),
                    reinterpret_cast<const char*>(&hash[1][i - 1]),
                    reinterpret_cast<const char*>(&hash[2][i - 1]),
                    reinterpret_cast<const char*>(&hash[3][i - 1]),
                    reinterpret_cast<char*>(&hash[0][i]),
                    reinterpret_cast<char*>(&hash[1][i]),
                    reinterpret_cast<char*>(&hash[2][i]),
                    reinterpret_cast<char*>(&hash[3][i]),
                    64U, 1, params.page_size, params.iterations, params.aes_rounds))
                return false;
            continue;
        }

        if (width >= 2U) {
            for (; lane + 1 < count; lane += 2) {
                hash[lane][i] = uint512{};
                hash[lane + 1][i] = uint512{};
                if (!yerbas_cn_hash_pair_2way(
                        reinterpret_cast<const char*>(&hash[lane][i - 1]),
                        reinterpret_cast<const char*>(&hash[lane + 1][i - 1]),
                        reinterpret_cast<char*>(&hash[lane][i]),
                        reinterpret_cast<char*>(&hash[lane + 1][i]),
                        64U, 1, params.page_size, params.iterations, params.aes_rounds))
                    return false;
            }
        }
        for (; lane < count; ++lane) {
            if (!reusable_cn_hash(hash[lane][i - 1], hash[lane][i], algorithm)) return false;
        }
    }

    for (std::size_t lane = 0; lane < count; ++lane) {
        const uint256 result = hash[lane][17].trim256();
        std::memcpy(outputs[lane].data(), result.begin(), outputs[lane].size());
    }
    return true;
}

bool reference_ready() noexcept
{
    return true;
}

} // namespace yerbas::ghostrider
