#include "cpu/cpu_worker_pool.h"
#include "ghostrider/ghostrider.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Header = std::array<std::uint8_t, 80>;

unsigned int integer(const char* text)
{
    std::size_t consumed = 0;
    const auto n = std::stoul(text, &consumed);
    if (consumed != std::string(text).size() || n == 0 || n > 256)
        throw std::invalid_argument("integer");
    return static_cast<unsigned int>(n);
}

std::map<unsigned int, Header> corpus()
{
    std::map<unsigned int, Header> headers;
    std::uint32_t state = 0x729b351dU;
    for (unsigned int attempt = 0; attempt < 10000 && headers.size() < 20; ++attempt) {
        Header h{};
        h[0] = 4;
        for (std::size_t i = 4; i < 76; ++i) {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            h[i] = static_cast<std::uint8_t>(state);
        }
        unsigned int mask = 0;
        for (auto stage : yerbas::ghostrider::stage_schedule_quiet({h.data(), h.size()}))
            if (stage & yerbas::ghostrider::kCryptoNightStageFlag)
                mask |= 1U << (stage & 0x7fU);
        headers.emplace(mask, h);
    }
    if (headers.size() != 20) throw std::runtime_error("Incomplete CN corpus");
    return headers;
}

void nonce(Header& h, std::uint32_t n)
{
    for (unsigned int b = 0; b < 4; ++b) h[76 + b] = static_cast<std::uint8_t>(n >> (8 * b));
}

struct MeasurementGuard {
    MeasurementGuard() { yerbas::cpu::set_tuning_measurement_mode(true); }
    ~MeasurementGuard() { yerbas::cpu::set_tuning_measurement_mode(false); }
};
} // namespace

int main(int argc, char** argv)
{
    using namespace yerbas;
    unsigned int workers = 0, width = 0;
    cpu::CnWidthPolicy widths{};
    std::string width_label;
    double seconds = 1;
    try {
        if (argc != 4) throw std::invalid_argument("arguments");
        workers = integer(argv[1]);
        const std::string requested = argv[2];
        if (requested.find(',') == std::string::npos) widths.fill(integer(argv[2]));
        else {
            std::istringstream values(requested);
            std::string value;
            unsigned int count = 0;
            while (std::getline(values, value, ',')) {
                if (count == widths.size()) throw std::invalid_argument("width count");
                widths[count++] = integer(value.c_str());
            }
            if (count != widths.size() || requested.back() == ',')
                throw std::invalid_argument("width count");
        }
        for (auto n : widths) {
            if (n != 1 && n != 2 && n != 4) throw std::invalid_argument("width");
            if (!width_label.empty()) width_label += '/';
            width_label += std::to_string(n);
            width = std::max(width, n);
        }
        std::size_t consumed = 0;
        seconds = std::stod(argv[3], &consumed);
        if (consumed != std::string(argv[3]).size() || !std::isfinite(seconds) ||
            seconds < 0.1 || seconds > 60)
            throw std::invalid_argument("range");
    } catch (...) {
        std::cerr << "Usage: cpu-plan-benchmark workers(1..256) CN-width(1|2|4 or six comma-separated widths) seconds-per-header(0.1..60)\n";
        return 1;
    }
    if (const char* policy = std::getenv("YERBAS_CPU_COMBO_POLICY_FILE"); policy && *policy) {
        std::cerr << "Unset YERBAS_CPU_COMBO_POLICY_FILE for an explicit width comparison\n";
        return 1;
    }
    try {
        const auto headers = corpus();
        MeasurementGuard guard;
        if (!ghostrider::optimized_cpu_ready()) throw std::runtime_error("CPU parity failed");
        // Check the actual grouped full-GhostRider route for every CN combination.
        for (const auto& entry : headers) {
            std::array<Header, 4> h{};
            std::array<ghostrider::Work, 4> works{};
            std::array<ghostrider::Hash256, 4> hashes{};
            for (unsigned int i = 0; i < width; ++i) {
                h[i] = entry.second; nonce(h[i], i);
                works[i] = {h[i].data(), h[i].size()};
            }
            if (!ghostrider::hash_optimized_batch(works.data(), hashes.data(), width, widths))
                throw std::runtime_error("Grouped CPU path unavailable");
            for (unsigned int i = 0; i < width; ++i)
                if (hashes[i] != ghostrider::hash_reference(works[i]))
                    throw std::runtime_error("Grouped full-GhostRider parity failed");
        }
        cpu::set_runtime_cn_widths(widths);
        cpu::WorkerPool pool(workers, width, cpu::AffinityPolicy::Unpinned);
        // 6x32 and 3x64 both cover exactly the same 192 nonces per timed batch.
        const unsigned int batch = ((192U + workers * width - 1U) / (workers * width)) * width;
        const std::uint32_t count = workers * batch;
        std::array<std::uint8_t, 32> target{}; target.fill(0xff);
        // Production pins the target to the header/job key. Use a separate
        // header key for the all-accepted nonce-coverage probe.
        auto coverage_header = headers.begin()->second;
        coverage_header[0] ^= 0x80U;
        auto found = pool.run(coverage_header, target, 0, batch, nullptr);
        std::vector<std::uint32_t> nonces;
        for (auto item : found) nonces.push_back(item.nonce);
        std::sort(nonces.begin(), nonces.end());
        if (nonces.size() != count) throw std::runtime_error("Worker hash count mismatch");
        for (std::uint32_t i = 0; i < count; ++i)
            if (nonces[i] != i) throw std::runtime_error("Worker nonce coverage mismatch");
        target.fill(0);
        std::cout << "CPU plan | corpus=v1 | combinations=20 | parity=PASS | workers=" << workers
                  << " | CN-widths=" << width_label << " | batch=" << batch
                  << " | affinity=unpinned | cache-writes=off\n" << std::flush;
        double equal_rotation_sum = 0;
        for (const auto& entry : headers) {
            (void)pool.run(entry.second, target, 0, width, nullptr);
            const auto start = std::chrono::steady_clock::now();
            std::uint64_t hashes = 0;
            double elapsed = 0;
            do {
                if (hashes + count > 0xffffffffULL) throw std::runtime_error("Nonce range exhausted");
                if (!pool.run(entry.second, target, static_cast<std::uint32_t>(hashes), batch, nullptr).empty())
                    throw std::runtime_error("Unexpected zero hash");
                hashes += count;
                elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            } while (elapsed < seconds);
            const double hps = hashes / elapsed;
            equal_rotation_sum += hps;
            std::cout << "mask=" << entry.first << " | hashes=" << hashes
                      << " | seconds=" << std::fixed << std::setprecision(6) << elapsed
                      << " | H/s=" << std::setprecision(2) << hps << '\n' << std::flush;
        }
        std::cout << "SUMMARY | workers=" << workers << " | CN-widths=" << width_label
                  << " | equal-rotation-mean=" << equal_rotation_sum / headers.size()
                  << " H/s | synthetic CPU-only; not hybrid/pool H/s\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
