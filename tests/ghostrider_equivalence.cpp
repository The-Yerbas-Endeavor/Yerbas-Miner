#include "ghostrider/ghostrider.h"

#include "hash_selection.h"
#include "uint256.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

namespace {

using yerbas::ghostrider::Hash256;
using yerbas::ghostrider::StageSchedule;
using yerbas::ghostrider::Work;

uint256 previous_block_hash(const Work& work)
{
    uint256 prev;
    std::memcpy(prev.begin(), work.data + 4, 32);
    return prev;
}

void legacy_selections(const Work& work,
                       std::vector<int>& core_hash_indexes,
                       std::vector<int>& random_cns)
{
    HashSelection selection(
        previous_block_hash(work),
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14},
        {0, 1, 2, 3, 4, 5});
    random_cns = selection.getCnIndexes();
    core_hash_indexes = selection.getAlgoIndexes();
}

StageSchedule legacy_schedule(const Work& work)
{
    std::vector<int> core;
    std::vector<int> cn;
    legacy_selections(work, core, cn);

    StageSchedule schedule{};
    for (int i = 0; i < 18; ++i) {
        if (i < 5) {
            schedule[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(core[i]);
        } else if (i == 5) {
            schedule[5] = static_cast<std::uint8_t>(yerbas::ghostrider::kCryptoNightStageFlag | cn[0]);
        } else if (i < 11) {
            schedule[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(core[i - 1]);
        } else if (i == 11) {
            schedule[11] = static_cast<std::uint8_t>(yerbas::ghostrider::kCryptoNightStageFlag | cn[1]);
        } else if (i < 17) {
            schedule[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(core[i - 2]);
        } else {
            schedule[17] = static_cast<std::uint8_t>(yerbas::ghostrider::kCryptoNightStageFlag | cn[2]);
        }
    }
    return schedule;
}

Hash256 legacy_hash(const Work& work)
{
    std::vector<int> core_hash_indexes;
    std::vector<int> random_cns;
    legacy_selections(work, core_hash_indexes, random_cns);

    // This intentionally preserves the exact execution order of the
    // last known-good Yerbas Core GhostRider path. Optimized implementations
    // must remain byte-for-byte equivalent to it.
    uint512 hash[18];
    for (int i = 0; i < 18; ++i) {
        const void* to_hash = i == 0
            ? static_cast<const void*>(work.data)
            : static_cast<const void*>(&hash[i - 1]);
        const int len_to_hash = i == 0 ? static_cast<int>(work.size) : 64;

        int core_selection = -1;
        int cn_selection = -1;
        if (i < 5) {
            core_selection = core_hash_indexes[i];
        } else if (i == 5) {
            cn_selection = random_cns[0];
        } else if (i < 11) {
            core_selection = core_hash_indexes[i - 1];
        } else if (i == 11) {
            cn_selection = random_cns[1];
        } else if (i < 17) {
            core_selection = core_hash_indexes[i - 2];
        } else {
            cn_selection = random_cns[2];
        }

        coreHash(to_hash, &hash[i], len_to_hash, core_selection);
        uint512* cn_input = i == 0 ? &hash[0] : &hash[i - 1];
        cnHash(cn_input, &hash[i], len_to_hash, cn_selection);
    }

    const uint256 result = hash[17].trim256();
    Hash256 out{};
    std::memcpy(out.data(), result.begin(), out.size());
    return out;
}

std::array<std::uint8_t, 80> make_header(std::uint32_t seed)
{
    std::array<std::uint8_t, 80> header{};
    std::uint32_t x = seed ^ 0x9e3779b9U;
    for (std::size_t i = 0; i < header.size(); ++i) {
        // Deterministic xorshift corpus: changing the previous-block bytes
        // exercises different GhostRider core/CryptoNight rotations.
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        header[i] = static_cast<std::uint8_t>(x >> 24);
    }

    // Use a conventional little-endian version field and vary the nonce too.
    header[0] = 0x00;
    header[1] = 0x00;
    header[2] = 0x00;
    header[3] = 0x20;
    header[76] = static_cast<std::uint8_t>(seed);
    header[77] = static_cast<std::uint8_t>(seed >> 8);
    header[78] = static_cast<std::uint8_t>(seed >> 16);
    header[79] = static_cast<std::uint8_t>(seed >> 24);
    return header;
}

std::string hex_hash(const Hash256& hash)
{
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (const auto byte : hash) out << std::setw(2) << static_cast<unsigned int>(byte);
    return out.str();
}

} // namespace

int main()
{
    constexpr std::size_t kCorpusSize = 96;
    std::size_t distinct_schedules = 0;
    std::vector<StageSchedule> seen_schedules;
    seen_schedules.reserve(kCorpusSize);

    for (std::size_t i = 0; i < kCorpusSize; ++i) {
        const auto header = make_header(static_cast<std::uint32_t>(0x1000U + i * 0x9e37U));
        const Work work{header.data(), header.size()};

        const auto expected_schedule = legacy_schedule(work);
        const auto actual_schedule = yerbas::ghostrider::stage_schedule(work);
        if (actual_schedule != expected_schedule) {
            std::cerr << "GhostRider schedule regression at corpus index " << i << '\n';
            return 1;
        }

        bool new_schedule = true;
        for (const auto& seen : seen_schedules) {
            if (seen == actual_schedule) {
                new_schedule = false;
                break;
            }
        }
        if (new_schedule) {
            seen_schedules.push_back(actual_schedule);
            ++distinct_schedules;
        }

        const auto expected_hash = legacy_hash(work);
        const auto actual_hash = yerbas::ghostrider::hash_reference(work);
        if (actual_hash != expected_hash) {
            std::cerr << "GhostRider hash regression at corpus index " << i << '\n'
                      << "expected: " << hex_hash(expected_hash) << '\n'
                      << "actual:   " << hex_hash(actual_hash) << '\n';
            return 2;
        }
    }

    // A rotating-algorithm regression corpus is only useful if it actually
    // exercises multiple schedules. This catches accidental corpus breakage.
    if (distinct_schedules < 16) {
        std::cerr << "GhostRider regression corpus has insufficient rotation coverage: "
                  << distinct_schedules << " distinct schedules\n";
        return 3;
    }

    std::cout << "GhostRider equivalence passed: " << kCorpusSize
              << " headers, " << distinct_schedules << " distinct schedules\n";
    return 0;
}
