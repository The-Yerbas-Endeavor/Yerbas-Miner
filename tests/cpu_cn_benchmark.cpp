#include "ghostrider/ghostrider.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

// Single-worker fixed-input screen. This is not full GhostRider or hybrid H/s.
// Run the same binary/input set in A/B/B/A order before judging a kernel change.
int main(int argc, char** argv)
{
    using namespace yerbas::ghostrider;
    double seconds = 2.0;
    try {
        if (argc > 2) throw std::invalid_argument("arguments");
        if (argc == 2) {
            std::size_t consumed = 0;
            seconds = std::stod(argv[1], &consumed);
            if (consumed != std::string(argv[1]).size())
                throw std::invalid_argument("seconds");
        }
        if (!std::isfinite(seconds) || seconds < 0.1 || seconds > 60.0)
            throw std::invalid_argument("range");
    } catch (...) {
        std::cerr << "Usage: cpu-cn-benchmark [seconds-per-variant: 0.1..60]\n";
        return 1;
    }

    std::cout << "CPU CN screen | workers=1 | input-set=v1 | seconds-per-variant="
              << seconds << " | not whole-miner H/s\n";
    for (std::uint8_t variant = 0; variant < 6; ++variant) {
        Hash512 input{}, output{};
        for (std::size_t i = 0; i < input.size(); ++i)
            input[i] = static_cast<std::uint8_t>(i * 17U + variant * 13U);

        // Exercise distinct input bytes against the pinned consensus reference
        // before timing. Keep this work out of the throughput measurement.
        for (unsigned int sample = 0; sample < 4; ++sample) {
            input[0] = static_cast<std::uint8_t>(sample);
            const auto reference = stage_reference(
                {input.data(), input.size()}, kCryptoNightStageFlag | variant);
            if (!optimized_cn_stage(input, variant, output) || output != reference) {
                std::cerr << "Parity FAIL: " << cryptonight_name(variant)
                          << " input=" << sample << '\n';
                return 2;
            }
        }

        const auto start = std::chrono::steady_clock::now();
        std::uint64_t hashes = 0, checksum = 0;
        double elapsed = 0;
        do {
            input[0] = static_cast<std::uint8_t>(hashes);
            if (!optimized_cn_stage(input, variant, output)) return 2;
            checksum += output[0];
            ++hashes;
            elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
        } while (elapsed < seconds);

        std::cout << cryptonight_name(variant) << " | parity=PASS | hashes="
                  << hashes << " | seconds=" << std::fixed << std::setprecision(6)
                  << elapsed << " | H/s=" << std::setprecision(2)
                  << static_cast<double>(hashes) / elapsed
                  << " | checksum=" << checksum << '\n';
    }
}
