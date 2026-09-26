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
#include <thread>
#include <unordered_map>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <termios.h>
#include <unistd.h>
#endif

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

inline std::mutex& session_stream_mutex()
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
        configure_input();
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
        restore_input();
    }

private:
    void configure_input() noexcept
    {
#ifdef _WIN32
        input_handle_ = GetStdHandle(STD_INPUT_HANDLE);
        if (input_handle_ == INVALID_HANDLE_VALUE || input_handle_ == nullptr)
            return;

        DWORD mode = 0;
        if (!GetConsoleMode(input_handle_, &mode))
            return;

        saved_input_mode_ = mode;
        input_mode_saved_ = true;

        // Keep ENABLE_PROCESSED_INPUT so Ctrl+C continues to generate the
        // normal console control event, but stop ordinary keys/mouse-wheel
        // escape sequences from being echoed into the dashboard.
        mode &= ~(ENABLE_ECHO_INPUT | ENABLE_LINE_INPUT);
        SetConsoleMode(input_handle_, mode);
        FlushConsoleInputBuffer(input_handle_);
#else
        if (tcgetattr(STDIN_FILENO, &saved_termios_) != 0)
            return;

        termios_saved_ = true;

        struct termios tui = saved_termios_;
        tui.c_lflag &= static_cast<tcflag_t>(~(ECHO | ECHONL));

        // Leave ISIG enabled so Ctrl+C keeps working normally. We do not make
        // the miner interactive here; this only prevents terminal input from
        // being visibly echoed over the alternate-screen dashboard.
        tcsetattr(STDIN_FILENO, TCSANOW, &tui);
        tcflush(STDIN_FILENO, TCIFLUSH);
#endif
    }

    void restore_input() noexcept
    {
#ifdef _WIN32
        if (!input_mode_saved_)
            return;

        FlushConsoleInputBuffer(input_handle_);
        SetConsoleMode(input_handle_, saved_input_mode_);
#else
        if (!termios_saved_)
            return;

        tcflush(STDIN_FILENO, TCIFLUSH);
        tcsetattr(STDIN_FILENO, TCSANOW, &saved_termios_);
#endif
    }

    bool active_{false};

#ifdef _WIN32
    HANDLE input_handle_{INVALID_HANDLE_VALUE};
    DWORD saved_input_mode_{0};
    bool input_mode_saved_{false};
#else
    struct termios saved_termios_ {};
    bool termios_saved_{false};
#endif
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
        write_buffered(&c, 1);
        return ch;
    }

    std::streamsize xsputn(const char* data, std::streamsize size) override
    {
        if (size <= 0) return 0;
        write_buffered(data, static_cast<std::size_t>(size));
        return size;
    }

    int sync() override
    {
        auto& pending = thread_buffer();
        if (!pending.empty()) {
            emit_chunk(pending);
            pending.clear();
        }

        std::lock_guard<std::mutex> lock(session_stream_mutex());

        int terminal_result = 0;
        if (!muteable_terminal_ || terminal_stdout_enabled())
            terminal_result = terminal_->pubsync();

        const int file_result = file_->pubsync();
        return (terminal_result == 0 && file_result == 0) ? 0 : -1;
    }

private:
    static thread_local std::unordered_map<const DirectMirrorBuf*, std::string> thread_buffers_;

    std::string& thread_buffer()
    {
        return thread_buffers_[this];
    }

    void write_buffered(const char* data, std::size_t size)
    {
        auto& pending = thread_buffer();
        pending.append(data, size);

        std::size_t newline = std::string::npos;
        while ((newline = pending.find('\n')) != std::string::npos) {
            std::string line = pending.substr(0, newline + 1U);
            pending.erase(0, newline + 1U);
            emit_chunk(line);
        }
    }

    void emit_chunk(const std::string& chunk)
    {
        std::lock_guard<std::mutex> lock(session_stream_mutex());

        if (!muteable_terminal_ || terminal_stdout_enabled())
            terminal_->sputn(
                chunk.data(),
                static_cast<std::streamsize>(chunk.size()));

        file_->sputn(
            chunk.data(),
            static_cast<std::streamsize>(chunk.size()));
    }

    std::streambuf* terminal_;
    std::streambuf* file_;
    bool muteable_terminal_{false};
};

inline thread_local std::unordered_map<const DirectMirrorBuf*, std::string>
    DirectMirrorBuf::thread_buffers_;

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
