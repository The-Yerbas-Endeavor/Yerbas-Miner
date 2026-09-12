#pragma once

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>

namespace yerbas::stratum {

// Lightweight production telemetry for GhostRider rotation fingerprints and
// ordered CryptoNight triples.
//
// The Stratum client already resets its rotation counters whenever the
// GhostRider schedule fingerprint changes. These wrappers preserve the
// existing counter semantics while accumulating both per-fingerprint and
// cross-fingerprint throughput for the same ordered CN triple. No device
// names, compute capabilities, or hardware-specific thresholds are used here:
// the measurements describe whatever hardware is actually mining.
class RotationTelemetryState {
public:
    struct Aggregate {
        std::uint64_t samples{0};
        std::uint64_t hashes{0};
        std::uint64_t cpu_hashes{0};
        double seconds{0.0};
        double ewma_hps{0.0};
    };

    static void label(std::uint64_t fingerprint, const std::string& ordered_cn)
    {
        if (fingerprint != 0 && !ordered_cn.empty()) labels_[fingerprint] = ordered_cn;
    }

    static void activate(std::uint64_t fingerprint)
    {
        const auto now = clock::now();
        if (active_fingerprint_ == fingerprint) return;

        finalize(now);
        active_fingerprint_ = fingerprint;
        active_started_ = now;
        active_hashes_ = 0;
        active_cpu_hashes_ = 0;
        const auto label_it = labels_.find(fingerprint);
        active_ordered_cn_ = label_it == labels_.end() ? std::string{} : label_it->second;
    }

    static void add_total(std::uint64_t hashes) noexcept
    {
        if (active_fingerprint_ != 0) active_hashes_ += hashes;
    }

    static void add_cpu(std::uint64_t hashes) noexcept
    {
        if (active_fingerprint_ != 0) active_cpu_hashes_ += hashes;
    }

private:
    using clock = std::chrono::steady_clock;

    static void add_sample(Aggregate& aggregate,
                           std::uint64_t hashes,
                           std::uint64_t cpu_hashes,
                           double seconds,
                           double total_hps)
    {
        ++aggregate.samples;
        aggregate.hashes += hashes;
        aggregate.cpu_hashes += cpu_hashes;
        aggregate.seconds += seconds;
        aggregate.ewma_hps = aggregate.samples == 1
            ? total_hps
            : (aggregate.ewma_hps * 0.75 + total_hps * 0.25);
    }

    static double lifetime_hps(const Aggregate& aggregate)
    {
        return aggregate.seconds > 0.0
            ? static_cast<double>(aggregate.hashes) / aggregate.seconds : 0.0;
    }

    static void finalize(clock::time_point now)
    {
        if (active_fingerprint_ == 0 || active_started_.time_since_epoch().count() == 0) return;

        const double seconds = std::chrono::duration<double>(now - active_started_).count();
        if (seconds <= 0.0 || active_hashes_ == 0) return;

        const std::uint64_t gpu_hashes = active_hashes_ >= active_cpu_hashes_
            ? active_hashes_ - active_cpu_hashes_ : 0;
        const double total_hps = static_cast<double>(active_hashes_) / seconds;
        const double cpu_hps = static_cast<double>(active_cpu_hashes_) / seconds;
        const double gpu_hps = static_cast<double>(gpu_hashes) / seconds;

        auto& fingerprint_aggregate = aggregates_[active_fingerprint_];
        add_sample(fingerprint_aggregate, active_hashes_, active_cpu_hashes_, seconds, total_hps);
        const double fingerprint_hps = lifetime_hps(fingerprint_aggregate);

        Aggregate* ordered_aggregate = nullptr;
        if (!active_ordered_cn_.empty()) {
            ordered_aggregate = &ordered_cn_aggregates_[active_ordered_cn_];
            add_sample(*ordered_aggregate, active_hashes_, active_cpu_hashes_, seconds, total_hps);
        }

        const auto flags = std::cout.flags();
        const auto precision = std::cout.precision();
        std::cout << "[rotation perf] fingerprint="
                  << std::hex << std::setfill('0') << std::setw(16) << active_fingerprint_
                  << std::dec << std::setfill(' ')
                  << " | sample=" << fingerprint_aggregate.samples
                  << " | duration=" << std::fixed << std::setprecision(2) << seconds << "s"
                  << " | hashes=" << active_hashes_
                  << " | total=" << std::setprecision(2) << total_hps << " H/s"
                  << " | cpu=" << cpu_hps << " H/s"
                  << " | gpu=" << gpu_hps << " H/s"
                  << " | fingerprint_avg=" << fingerprint_hps << " H/s"
                  << " | ewma=" << fingerprint_aggregate.ewma_hps << " H/s";
        if (ordered_aggregate) {
            std::cout << " | cn=" << active_ordered_cn_
                      << " | cn_samples=" << ordered_aggregate->samples
                      << " | cn_avg=" << lifetime_hps(*ordered_aggregate) << " H/s"
                      << " | cn_ewma=" << ordered_aggregate->ewma_hps << " H/s";
        }
        std::cout << '\n';
        std::cout.flags(flags);
        std::cout.precision(precision);
    }

    inline static std::uint64_t active_fingerprint_{0};
    inline static clock::time_point active_started_{};
    inline static std::uint64_t active_hashes_{0};
    inline static std::uint64_t active_cpu_hashes_{0};
    inline static std::string active_ordered_cn_{};
    inline static std::unordered_map<std::uint64_t, std::string> labels_{};
    inline static std::unordered_map<std::uint64_t, Aggregate> aggregates_{};
    inline static std::unordered_map<std::string, Aggregate> ordered_cn_aggregates_{};
};

class RotationFingerprint {
public:
    RotationFingerprint() = default;
    RotationFingerprint(std::uint64_t value) : value_(value) {}

    RotationFingerprint& operator=(std::uint64_t value)
    {
        if (value_ != value) RotationTelemetryState::activate(value);
        value_ = value;
        return *this;
    }

    operator std::uint64_t() const noexcept { return value_; }

private:
    std::uint64_t value_{0};
};

class RotationHashCounter {
public:
    enum class Kind { Total, Cpu };

    explicit RotationHashCounter(Kind kind = Kind::Total) : kind_(kind) {}

    RotationHashCounter& operator=(std::uint64_t value) noexcept
    {
        value_ = value;
        return *this;
    }

    RotationHashCounter& operator+=(std::uint64_t value) noexcept
    {
        value_ += value;
        if (kind_ == Kind::Cpu) RotationTelemetryState::add_cpu(value);
        else RotationTelemetryState::add_total(value);
        return *this;
    }

    RotationHashCounter& operator++() noexcept
    {
        return (*this += 1);
    }

    operator std::uint64_t() const noexcept { return value_; }

private:
    Kind kind_{Kind::Total};
    std::uint64_t value_{0};
};

} // namespace yerbas::stratum
