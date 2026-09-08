#pragma once

#include <cstdlib>
#include <iostream>
#include <streambuf>
#include <string>

namespace yerbas::console {
namespace production_detail {

inline bool diagnostics_enabled()
{
    const char* value = std::getenv("YERBAS_DIAGNOSTICS");
    return value != nullptr && *value != '\0' && std::string(value) != "0";
}

inline bool suppress_line(const std::string& line)
{
    if (diagnostics_enabled()) return false;

    // CPU tuning progress stays concise in production: the existing
    // GhostRider-search start, final-validation and selected/cached summaries
    // remain visible, while individual measurements stay diagnostic-only.
    if (line.rfind("[CPU topology]", 0) == 0) return true;
    if (line.rfind("[CPU CN probe]", 0) == 0) return true;
    if (line.rfind("[CPU CN confirm]", 0) == 0) return true;
    if (line.rfind("[CPU affinity]", 0) == 0) return true;
    if (line.rfind("[CPU tune] baseline", 0) == 0) return true;

    // Per-rotation learning/probe output is intentionally retained for
    // diagnostics/perf capture but is far too noisy for the normal miner UI.
    if (line.rfind("[CPU fingerprint]", 0) == 0) return true;
    if (line.rfind("[GhostRider] rotation=", 0) == 0) return true;

    // Routine job lifecycle details do not require operator attention. Keep
    // share results, rejects, block finds, warnings and failures visible.
    if (line.rfind("[stratum] New job #", 0) == 0) return true;
    if (line.find("[hybrid] stale candidates suppressed") != std::string::npos) return true;
    if (line.rfind("[hybrid] Job partitioned:", 0) == 0) return true;

    // Rotation-adaptive GPU batch changes happen frequently and are tuning
    // telemetry rather than production status information.
    if (line.find("rotation-adaptive batch") != std::string::npos) return true;

    // CUDA selector/tuner details are useful for diagnostics and CSV capture,
    // but are unnecessary in the normal production console. Concise GPU
    // initialization/cache summaries emitted elsewhere remain visible.
    if (line.find("[CUDA CN stagger tuner]") != std::string::npos) return true;
    if (line.find("[CUDA CN stagger]") != std::string::npos) return true;
    if (line.find("[CUDA CN selector]") != std::string::npos) return true;
    if (line.find("[CUDA CN hardened selector]") != std::string::npos) return true;
    if (line.find("[CUDA CN validation]") != std::string::npos) return true;
    if (line.find("[CUDA CN parity]") != std::string::npos) return true;

    return false;
}

class ProductionLineBuf final : public std::streambuf {
public:
    explicit ProductionLineBuf(std::streambuf* destination) : destination_(destination) {}

protected:
    int overflow(int ch) override
    {
        if (ch == traits_type::eof()) return traits_type::not_eof(ch);
        const char c = static_cast<char>(ch);
        if (c == '\n') emit(true);
        else pending_.push_back(c);
        return ch;
    }

    std::streamsize xsputn(const char* s, std::streamsize count) override
    {
        for (std::streamsize i = 0; i < count; ++i) {
            if (s[i] == '\n') emit(true);
            else pending_.push_back(s[i]);
        }
        return count;
    }

    int sync() override
    {
        if (!pending_.empty()) emit(false);
        return destination_->pubsync();
    }

private:
    void emit(bool newline)
    {
        const bool suppressed = suppress_line(pending_);
        if (!suppressed && !pending_.empty())
            destination_->sputn(pending_.data(), static_cast<std::streamsize>(pending_.size()));
        if (!suppressed && newline) destination_->sputc('\n');
        pending_.clear();
    }

    std::streambuf* destination_{nullptr};
    std::string pending_;
};

} // namespace production_detail

inline void enable_production_output()
{
    static bool initialized = false;
    if (initialized) return;
    initialized = true;
    static production_detail::ProductionLineBuf production_cout(std::cout.rdbuf());
    std::cout.rdbuf(&production_cout);
}

} // namespace yerbas::console
