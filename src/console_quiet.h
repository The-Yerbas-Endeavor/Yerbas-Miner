#pragma once

#include <cstdlib>
#include <iostream>
#include <streambuf>
#include <string>

namespace yerbas::console {
namespace quiet_detail {

inline bool diagnostics_enabled()
{
    const char* value = std::getenv("YERBAS_DIAGNOSTICS");
    return value != nullptr && *value != '\0' && std::string(value) != "0";
}

inline bool suppress_normal_line(const std::string& line)
{
    // Per-batch performance instrumentation is useful while profiling but is
    // extremely noisy during normal pool mining. The Proof of Grass status
    // report already carries the useful sustained hashrate information.
    if (line.find("[CPU GR perf]") != std::string::npos) return true;
    if (line.find("[CUDA GR perf]") != std::string::npos) return true;
    if (line.find("[CUDA stage]") != std::string::npos) return true;
    if (line.find("GhostRider rolling stage profile") != std::string::npos) return true;
    if (line.find("[CUDA CN runtime]") != std::string::npos) return true;

    // Candidate discovery is immediately followed by SHARE SUBMITTED, so the
    // candidate line duplicates information without adding operational value.
    if (line.find("] candidate | job=") != std::string::npos) return true;

    // Detailed startup profiler lines stay available in diagnostics mode. Keep
    // compact CUDA autotune winner summaries visible in normal mode.
    if (line.find("[CUDA CN profile]") != std::string::npos) return true;
    if (line.find("[CPU CN profile]") != std::string::npos) return true;
    if (line.find("[CPU CN phase]") != std::string::npos) return true;
    if (line.find("[CPU CN candidate]") != std::string::npos) return true;
    if (line.find("[CPU CN A/B]") != std::string::npos) return true;
    if (line.find("[CUDA CN production test]") != std::string::npos) return true;
    if (line.find("[CUDA CN coop test]") != std::string::npos) return true;
    if (line.find("[CUDA dual-state test]") != std::string::npos) return true;

    // Cache filenames are implementation details and can fill most of startup
    // on multi-GPU systems. Selection summaries remain visible.
    if (line.find("CryptoNight geometry cache loaded |") != std::string::npos) return true;
    if (line.find("CryptoNight loop cache loaded |") != std::string::npos) return true;
    if (line.find("CryptoNight loop AES cache loaded |") != std::string::npos) return true;
    if (line.find("CryptoNight dual-state cache loaded |") != std::string::npos) return true;
    if (line.find("CryptoNight production loop cache loaded |") != std::string::npos) return true;
    if (line.find("CryptoNight cooperative cache loaded |") != std::string::npos) return true;

    // Coverage is already summarized by the compact cores 15/15 | CN 6/6 line.
    if (line.rfind("CUDA-ready cores:", 0) == 0) return true;
    if (line.rfind("CUDA pending cores:", 0) == 0) return true;
    if (line.rfind("CUDA pending CryptoNight:", 0) == 0) return true;

    return false;
}

class QuietLineBuf final : public std::streambuf {
public:
    explicit QuietLineBuf(std::streambuf* destination) : destination_(destination) {}

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
        const bool suppress = !diagnostics_enabled() && suppress_normal_line(pending_);
        if (!suppress && !pending_.empty())
            destination_->sputn(pending_.data(), static_cast<std::streamsize>(pending_.size()));
        if (!suppress && newline) destination_->sputc('\n');
        pending_.clear();
    }

    std::streambuf* destination_{nullptr};
    std::string pending_;
};

} // namespace quiet_detail

inline void enable_quiet_output()
{
    static bool initialized = false;
    if (initialized) return;
    initialized = true;
    static quiet_detail::QuietLineBuf quiet_cout(std::cout.rdbuf());
    std::cout.rdbuf(&quiet_cout);
}

} // namespace yerbas::console
