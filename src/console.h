#pragma once

#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <streambuf>
#include <string>
#include <unordered_map>

#ifdef _WIN32
#define NOMINMAX
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace yerbas::console {

namespace detail {

constexpr const char* kReset = "\x1b[0m";
constexpr const char* kDim = "\x1b[2m";
constexpr const char* kBold = "\x1b[1m";
constexpr const char* kRed = "\x1b[91m";
constexpr const char* kGreen = "\x1b[92m";
constexpr const char* kYellow = "\x1b[93m";
constexpr const char* kBlue = "\x1b[94m";
constexpr const char* kMagenta = "\x1b[95m";
constexpr const char* kCyan = "\x1b[96m";
constexpr const char* kWhite = "\x1b[97m";

inline bool terminal_supports_color()
{
    if (std::getenv("NO_COLOR") != nullptr) return false;
#ifdef _WIN32
    if (_isatty(_fileno(stdout)) == 0) return false;
    HANDLE handle = GetStdHandle(STD_OUTPUT_HANDLE);
    if (handle == INVALID_HANDLE_VALUE || handle == nullptr) return false;
    DWORD mode = 0;
    if (!GetConsoleMode(handle, &mode)) return false;
    if (!SetConsoleMode(handle, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) return false;
    return true;
#else
    return isatty(STDOUT_FILENO) != 0;
#endif
}

inline const char* color_for_line(const std::string& line)
{
    if (line.find("[share] ACCEPTED") != std::string::npos) return kGreen;
    if (line.find("[share] REJECTED") != std::string::npos ||
        line.find("Fatal:") != std::string::npos ||
        line.find(" failed") != std::string::npos ||
        line.find(" rejected") != std::string::npos) return kRed;
    if (line.rfind("[share]", 0) == 0) return kYellow;
    if (line.rfind("[GPU ", 0) == 0) {
        if (line.find("idle") != std::string::npos || line.find("incomplete") != std::string::npos) return kYellow;
        return kGreen;
    }
    if (line.rfind("[CPU]", 0) == 0) return kCyan;
    if (line.rfind("[TOTAL]", 0) == 0) return kWhite;
    if (line.rfind("[stratum]", 0) == 0) return kBlue;
    if (line.rfind("[hybrid]", 0) == 0) return kMagenta;
    if (line.find("CUDA") != std::string::npos) return kMagenta;
    if (line.find("warning") != std::string::npos || line.find("Warning") != std::string::npos) return kYellow;
    if (line.find("Yerbas Miner") != std::string::npos) return kBold;
    return nullptr;
}

struct GpuPowerSnapshot {
    std::string summary;
    double total_watts{0.0};
    bool available{false};
};

inline GpuPowerSnapshot query_gpu_power()
{
    GpuPowerSnapshot result;
#ifdef _WIN32
    FILE* pipe = _popen("nvidia-smi --query-gpu=index,power.draw --format=csv,noheader,nounits 2>NUL", "r");
#else
    FILE* pipe = popen("nvidia-smi --query-gpu=index,power.draw --format=csv,noheader,nounits 2>/dev/null", "r");
#endif
    if (pipe == nullptr) return result;

    char buffer[256];
    std::ostringstream summary;
    bool first = true;
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        std::string row(buffer);
        while (!row.empty() && (row.back() == '\n' || row.back() == '\r')) row.pop_back();
        const std::size_t comma = row.find(',');
        if (comma == std::string::npos) continue;
        std::string id = row.substr(0, comma);
        std::string watts = row.substr(comma + 1);
        const std::size_t first_nonspace = watts.find_first_not_of(" \t");
        if (first_nonspace == std::string::npos) continue;
        watts.erase(0, first_nonspace);
        try {
            const double value = std::stod(watts);
            if (!first) summary << "  ";
            summary << "GPU " << id << ' ' << std::fixed << std::setprecision(1) << value << " W";
            result.total_watts += value;
            result.available = true;
            first = false;
        } catch (...) {
        }
    }
#ifdef _WIN32
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    if (result.available) {
        summary << "  | TOTAL GPU " << std::fixed << std::setprecision(1) << result.total_watts << " W";
        result.summary = summary.str();
    }
    return result;
}

class LineColorBuf final : public std::streambuf {
public:
    LineColorBuf(std::streambuf* destination, bool enabled)
        : destination_(destination), enabled_(enabled)
    {
    }

protected:
    int overflow(int ch) override
    {
        if (ch == traits_type::eof()) return traits_type::not_eof(ch);
        auto& pending = thread_pending()[this];
        const char c = static_cast<char>(ch);
        if (c == '\n') {
            emit_line(pending, true);
        } else {
            pending.push_back(c);
        }
        return ch;
    }

    std::streamsize xsputn(const char* s, std::streamsize count) override
    {
        auto& pending = thread_pending()[this];
        for (std::streamsize i = 0; i < count; ++i) {
            if (s[i] == '\n') emit_line(pending, true);
            else pending.push_back(s[i]);
        }
        return count;
    }

    int sync() override
    {
        auto& pending = thread_pending()[this];
        if (!pending.empty()) emit_line(pending, false);
        std::lock_guard<std::mutex> lock(output_mutex());
        return destination_->pubsync();
    }

private:
    static std::unordered_map<const LineColorBuf*, std::string>& thread_pending()
    {
        static thread_local std::unordered_map<const LineColorBuf*, std::string> buffers;
        return buffers;
    }

    static std::mutex& output_mutex()
    {
        static std::mutex mutex;
        return mutex;
    }

    static std::string& rotation_fingerprint()
    {
        static std::string value;
        return value;
    }

    static std::string& rotation_cn_variants()
    {
        static std::string value;
        return value;
    }

    static void capture_rotation_metadata(const std::string& line)
    {
        constexpr const char* kFingerprintPrefix = "[GhostRider] schedule fingerprint=";
        if (line.rfind(kFingerprintPrefix, 0) == 0) {
            const std::size_t begin = std::char_traits<char>::length(kFingerprintPrefix);
            const std::size_t end = line.find_first_of(" |", begin);
            rotation_fingerprint() = line.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
            return;
        }

        constexpr const char* kSchedulePrefix = "[GhostRider] schedule ";
        if (line.rfind(kSchedulePrefix, 0) != 0 || line.rfind(kFingerprintPrefix, 0) == 0) return;

        std::string variants;
        std::size_t pos = std::char_traits<char>::length(kSchedulePrefix);
        while (pos < line.size()) {
            while (pos < line.size() && line[pos] == ' ') ++pos;
            if (pos >= line.size()) break;
            const std::size_t token_end = line.find(' ', pos);
            const std::string token = line.substr(pos, token_end == std::string::npos ? std::string::npos : token_end - pos);
            const std::size_t colon = token.find(':');
            if (colon != std::string::npos && token.compare(colon + 1, 3, "CN-") == 0) {
                if (!variants.empty()) variants += " / ";
                variants += token.substr(colon + 1);
            }
            if (token_end == std::string::npos) break;
            pos = token_end + 1;
        }
        if (!variants.empty()) rotation_cn_variants() = variants;
    }

    void write_raw_line(const std::string& line)
    {
        if (!line.empty()) destination_->sputn(line.data(), static_cast<std::streamsize>(line.size()));
        destination_->sputc('\n');
    }

    void emit_line(std::string& pending, bool newline)
    {
        std::lock_guard<std::mutex> lock(output_mutex());
        capture_rotation_metadata(pending);

        const char* color = enabled_ ? color_for_line(pending) : nullptr;
        if (color != nullptr) destination_->sputn(color, static_cast<std::streamsize>(std::char_traits<char>::length(color)));
        if (!pending.empty()) destination_->sputn(pending.data(), static_cast<std::streamsize>(pending.size()));
        if (color != nullptr) destination_->sputn(kReset, 4);
        if (newline) destination_->sputc('\n');

        if (newline && pending.find("PROOF OF GRASS | STATUS UPDATE") != std::string::npos) {
            if (!rotation_fingerprint().empty()) write_raw_line("ROTATION      " + rotation_fingerprint());
            if (!rotation_cn_variants().empty()) write_raw_line("CRYPTONIGHT   " + rotation_cn_variants());
            const auto power = query_gpu_power();
            write_raw_line(power.available ? "POWER         " + power.summary : "POWER         GPU power unavailable");
        }
        pending.clear();
    }

    std::streambuf* destination_{nullptr};
    bool enabled_{false};
};

} // namespace detail

inline void enable_colors()
{
    static bool initialized = false;
    if (initialized) return;
    initialized = true;

    const bool enabled = detail::terminal_supports_color();
    static detail::LineColorBuf cout_buffer(std::cout.rdbuf(), enabled);
    static detail::LineColorBuf cerr_buffer(std::cerr.rdbuf(), enabled);
    std::cout.rdbuf(&cout_buffer);
    std::cerr.rdbuf(&cerr_buffer);
}

} // namespace yerbas::console
