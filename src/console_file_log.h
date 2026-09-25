#pragma once

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <streambuf>
#include <string>

namespace yerbas::console {

inline std::streambuf*& native_terminal_stdout()
{
    static std::streambuf* value = nullptr;
    return value;
}

inline bool& terminal_stdout_enabled()
{
    static bool value = true;
    return value;
}

inline bool& dashboard_active_flag()
{
    static bool value = false;
    return value;
}

inline std::mutex& terminal_write_mutex()
{
    static std::mutex value;
    return value;
}

inline bool dashboard_active() noexcept
{
    return dashboard_active_flag();
}

inline void terminal_write(const std::string& value)
{
    std::lock_guard<std::mutex> lock(terminal_write_mutex());
    std::streambuf* out = native_terminal_stdout();
    if (out == nullptr) return;
    out->sputn(value.data(), static_cast<std::streamsize>(value.size()));
    out->pubsync();
}

class DashboardScreen final {
public:
    DashboardScreen()
    {
        dashboard_active_flag() = true;
        terminal_stdout_enabled() = false;
        terminal_write("\x1b[?1049h\x1b[?25l\x1b[2J\x1b[H");
        active_ = true;
    }

    DashboardScreen(const DashboardScreen&) = delete;
    DashboardScreen& operator=(const DashboardScreen&) = delete;

    ~DashboardScreen() noexcept
    {
        if (!active_) return;
        try {
            terminal_write("\x1b[0m\x1b[?25h\x1b[?1049l");
        } catch (...) {}
        dashboard_active_flag() = false;
        terminal_stdout_enabled() = true;
    }

private:
    bool active_{false};
};

class DirectMirrorBuf final : public std::streambuf {
public:
    DirectMirrorBuf(std::streambuf* terminal,
                    std::streambuf* file,
                    bool muteable_terminal) noexcept
        : terminal_(terminal),
          file_(file),
          muteable_terminal_(muteable_terminal) {}

protected:
    int_type overflow(int_type ch) override
    {
        if (traits_type::eq_int_type(ch, traits_type::eof()))
            return traits_type::not_eof(ch);

        const char c = traits_type::to_char_type(ch);

        bool terminal_ok = true;
        if (!muteable_terminal_ || terminal_stdout_enabled()) {
            const auto a = terminal_->sputc(c);
            terminal_ok = !traits_type::eq_int_type(a, traits_type::eof());
        }

        const auto b = file_->sputc(c);
        const bool file_ok = !traits_type::eq_int_type(b, traits_type::eof());

        return (terminal_ok && file_ok) ? ch : traits_type::eof();
    }

    std::streamsize xsputn(const char* data, std::streamsize size) override
    {
        std::streamsize terminal_written = size;
        if (!muteable_terminal_ || terminal_stdout_enabled())
            terminal_written = terminal_->sputn(data, size);

        const auto file_written = file_->sputn(data, size);
        return terminal_written < file_written ? terminal_written : file_written;
    }

    int sync() override
    {
        int terminal_result = 0;
        if (!muteable_terminal_ || terminal_stdout_enabled())
            terminal_result = terminal_->pubsync();

        const int file_result = file_->pubsync();
        return (terminal_result == 0 && file_result == 0) ? 0 : -1;
    }

private:
    std::streambuf* terminal_;
    std::streambuf* file_;
    bool muteable_terminal_{false};
};

class SessionFileLog final {
public:
    explicit SessionFileLog(const std::string& path)
    {
        if (native_terminal_stdout() == nullptr)
            native_terminal_stdout() = std::cout.rdbuf();

        if (path.empty()) return;

        std::error_code ec;
        const std::filesystem::path fs_path(path);
        if (fs_path.has_parent_path())
            std::filesystem::create_directories(fs_path.parent_path(), ec);

        file_.open(path, std::ios::out | std::ios::app);
        if (!file_) return;

        cout_native_ = std::cout.rdbuf();
        cerr_native_ = std::cerr.rdbuf();
        cout_mirror_ = std::make_unique<DirectMirrorBuf>(cout_native_, file_.rdbuf(), true);
        // While the alternate-screen dashboard is active, stderr is logged but
        // not allowed to scribble over the TUI. Fatal startup/exit errors still
        // appear normally after DashboardScreen restores the terminal.
        cerr_mirror_ = std::make_unique<DirectMirrorBuf>(cerr_native_, file_.rdbuf(), true);
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

inline std::string default_session_log_path()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t stamp = std::chrono::system_clock::to_time_t(now);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &stamp);
#else
    localtime_r(&stamp, &local);
#endif
    std::ostringstream path;
    path << "logs/yerbas-miner-"
         << std::put_time(&local, "%Y%m%d-%H%M%S")
         << ".log";
    return path.str();
}

inline std::string session_log_path_from_env()
{
    const char* value = std::getenv("YERBAS_LOG_FILE");
    if (value != nullptr && *value != '\0') {
        const std::string requested(value);
        if (requested == "0" || requested == "off" ||
            requested == "false" || requested == "no")
            return {};
        return requested;
    }
    return default_session_log_path();
}

} // namespace yerbas::console
