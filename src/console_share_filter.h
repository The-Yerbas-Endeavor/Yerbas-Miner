#pragma once

#include <cstdlib>
#include <iostream>
#include <streambuf>
#include <string>

namespace yerbas::console {
namespace share_filter_detail {

inline bool verbose_shares_enabled()
{
    const char* value = std::getenv("YERBAS_VERBOSE_SHARES");
    if (value == nullptr || *value == '\0') return false;
    const std::string v(value);
    return v != "0" && v != "off" && v != "false";
}

inline bool suppress_line(const std::string& line)
{
    if (verbose_shares_enabled()) return false;

    // High-frequency submit/accept chatter is redundant with the periodic
    // status table and can overwhelm terminal rendering on fast miners.
    // Rejections and block-found messages intentionally remain visible.
    if (line.find("SHARE SUBMITTED") != std::string::npos) return true;
    if (line.find("SHARE ACCEPTED") != std::string::npos) return true;
    if (line.find("[share] ACCEPTED") != std::string::npos) return true;
    return false;
}

class ShareFilterBuf final : public std::streambuf {
public:
    explicit ShareFilterBuf(std::streambuf* destination) : destination_(destination) {}

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
        if (!suppress_line(pending_)) {
            if (!pending_.empty())
                destination_->sputn(pending_.data(), static_cast<std::streamsize>(pending_.size()));
            if (newline) destination_->sputc('\n');
        }
        pending_.clear();
    }

    std::streambuf* destination_{nullptr};
    std::string pending_;
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
