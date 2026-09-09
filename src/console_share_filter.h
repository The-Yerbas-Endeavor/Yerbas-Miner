#pragma once

#include <cstdlib>
#include <iostream>
#include <mutex>
#include <streambuf>
#include <string>

namespace yerbas::console {
namespace share_filter_detail {

inline bool env_enabled(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') return false;
    const std::string v(value);
    return v != "0" && v != "off" && v != "false";
}

inline bool verbose_shares_enabled()
{
    return env_enabled("YERBAS_VERBOSE_SHARES");
}

inline bool diagnostics_enabled()
{
    return env_enabled("YERBAS_DIAGNOSTICS");
}

inline bool suppress_line(const std::string& line)
{
    if (!verbose_shares_enabled()) {
        // High-frequency submit/accept chatter is redundant with the periodic
        // status table and can overwhelm terminal rendering on fast miners.
        // Rejections and block-found messages intentionally remain visible.
        if (line.find("SHARE SUBMITTED") != std::string::npos) return true;
        if (line.find("SHARE ACCEPTED") != std::string::npos) return true;
        if (line.find("[share] ACCEPTED") != std::string::npos) return true;
    }

    if (!diagnostics_enabled()) {
        // Runtime tuning/profiling details are useful in diagnostic captures,
        // but they are both noisy and frequent during normal production.
        // Keep the user-facing rotation, status, reject, block and connection
        // messages while hiding internal calibration chatter.
        if (line.find("[CPU fingerprint]") != std::string::npos) return true;
        if (line.find("CryptoNight phase backend cache loaded") != std::string::npos) return true;
        if (line.find("CryptoNight production selector cache loaded") != std::string::npos) return true;
        if (line.find("CryptoNight production geometry cache loaded") != std::string::npos) return true;
        if (line.find("[CUDA CN production geometry]") != std::string::npos) return true;
        if (line.find("[CUDA CN geometry sample]") != std::string::npos) return true;
        if (line.find("[CUDA CN geometry result]") != std::string::npos) return true;
        if (line.find("[CUDA CN geometry selected]") != std::string::npos) return true;
    }

    return false;
}

class ShareFilterBuf final : public std::streambuf {
public:
    explicit ShareFilterBuf(std::streambuf* destination) : destination_(destination) {}

protected:
    int overflow(int ch) override
    {
        if (ch == traits_type::eof()) return traits_type::not_eof(ch);
        auto& pending = thread_pending();
        const char c = static_cast<char>(ch);
        if (c == '\n') emit(pending, true);
        else pending.push_back(c);
        return ch;
    }

    std::streamsize xsputn(const char* s, std::streamsize count) override
    {
        auto& pending = thread_pending();
        for (std::streamsize i = 0; i < count; ++i) {
            if (s[i] == '\n') emit(pending, true);
            else pending.push_back(s[i]);
        }
        return count;
    }

    int sync() override
    {
        auto& pending = thread_pending();
        if (!pending.empty()) emit(pending, false);
        std::lock_guard<std::mutex> lock(output_mutex_);
        return destination_->pubsync();
    }

private:
    static std::string& thread_pending()
    {
        // std::cout is written concurrently by CPU workers, GPU workers and
        // Stratum/status code. A shared line buffer lets fragments from those
        // threads corrupt each other. Build each thread's line independently,
        // then serialize only the completed line to the downstream filters.
        static thread_local std::string pending;
        return pending;
    }

    void emit(std::string& pending, bool newline)
    {
        std::string line;
        line.swap(pending);
        if (suppress_line(line)) return;

        std::lock_guard<std::mutex> lock(output_mutex_);
        if (!line.empty())
            destination_->sputn(line.data(), static_cast<std::streamsize>(line.size()));
        if (newline) destination_->sputc('\n');
    }

    std::streambuf* destination_{nullptr};
    std::mutex output_mutex_;
};

} // namespace share_filter_detail

inline void enable_share_filter()
{
    static bool initialized = false;
    if (initialized) return;
    initialized = true;
    static share_filter_detail::ShareFilterBuf share_filter(std::cout.rdbuf());
    std::cout.rdbuf(&share_filter);
}

} // namespace yerbas::console
