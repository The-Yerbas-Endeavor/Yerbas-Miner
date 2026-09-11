#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <streambuf>
#include <string>

namespace yerbas::console {

class DirectMirrorBuf final : public std::streambuf {
public:
    DirectMirrorBuf(std::streambuf* terminal, std::streambuf* file) noexcept
        : terminal_(terminal), file_(file) {}

protected:
    int_type overflow(int_type ch) override
    {
        if (traits_type::eq_int_type(ch, traits_type::eof()))
            return traits_type::not_eof(ch);

        const char c = traits_type::to_char_type(ch);
        const auto a = terminal_->sputc(c);
        const auto b = file_->sputc(c);
        return (traits_type::eq_int_type(a, traits_type::eof()) ||
                traits_type::eq_int_type(b, traits_type::eof()))
                   ? traits_type::eof()
                   : ch;
    }

    std::streamsize xsputn(const char* data, std::streamsize size) override
    {
        const auto a = terminal_->sputn(data, size);
        const auto b = file_->sputn(data, size);
        return a < b ? a : b;
    }

    int sync() override
    {
        const int a = terminal_->pubsync();
        const int b = file_->pubsync();
        return (a == 0 && b == 0) ? 0 : -1;
    }

private:
    std::streambuf* terminal_;
    std::streambuf* file_;
};

class SessionFileLog final {
public:
    explicit SessionFileLog(const std::string& path)
    {
        if (path.empty()) return;

        std::error_code ec;
        const std::filesystem::path fs_path(path);
        if (fs_path.has_parent_path())
            std::filesystem::create_directories(fs_path.parent_path(), ec);

        file_.open(path, std::ios::out | std::ios::app);
        if (!file_) return;

        cout_native_ = std::cout.rdbuf();
        cerr_native_ = std::cerr.rdbuf();
        cout_mirror_ = std::make_unique<DirectMirrorBuf>(cout_native_, file_.rdbuf());
        cerr_mirror_ = std::make_unique<DirectMirrorBuf>(cerr_native_, file_.rdbuf());
        std::cout.rdbuf(cout_mirror_.get());
        std::cerr.rdbuf(cerr_mirror_.get());
        active_ = true;
    }

    SessionFileLog(const SessionFileLog&) = delete;
    SessionFileLog& operator=(const SessionFileLog&) = delete;

    ~SessionFileLog() noexcept
    {
        if (!active_) return;
        try { std::cout.flush(); } catch (...) {}
        try { std::cerr.flush(); } catch (...) {}
        std::cout.rdbuf(cout_native_);
        std::cerr.rdbuf(cerr_native_);
        try { file_.flush(); } catch (...) {}
    }

    bool active() const noexcept { return active_; }

private:
    std::ofstream file_;
    std::streambuf* cout_native_{nullptr};
    std::streambuf* cerr_native_{nullptr};
    std::unique_ptr<DirectMirrorBuf> cout_mirror_;
    std::unique_ptr<DirectMirrorBuf> cerr_mirror_;
    bool active_{false};
};

inline std::string session_log_path_from_env()
{
    const char* value = std::getenv("YERBAS_LOG_FILE");
    return (value != nullptr && *value != '\0') ? std::string(value) : std::string{};
}

} // namespace yerbas::console
