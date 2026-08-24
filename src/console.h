#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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
    if (line.find("[share] REJECTED") != std::string::npos || line.find("Fatal:") != std::string::npos ||
        line.find(" failed") != std::string::npos || line.find(" rejected") != std::string::npos) return kRed;
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

struct GpuTelemetry {
    double watts{0.0};
    double temperature_c{0.0};
    double fan_percent{0.0};
    bool power_available{false};
    bool temperature_available{false};
    bool fan_available{false};
};

struct GpuTelemetrySnapshot {
    std::unordered_map<int, GpuTelemetry> devices;
    bool available{false};
};

inline GpuTelemetrySnapshot query_gpu_telemetry()
{
    GpuTelemetrySnapshot result;
#ifdef _WIN32
    FILE* pipe = _popen("nvidia-smi --query-gpu=index,power.draw,temperature.gpu,fan.speed --format=csv,noheader,nounits 2>NUL", "r");
#else
    FILE* pipe = popen("nvidia-smi --query-gpu=index,power.draw,temperature.gpu,fan.speed --format=csv,noheader,nounits 2>/dev/null", "r");
#endif
    if (pipe == nullptr) return result;

    char buffer[256];
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        std::string row(buffer);
        while (!row.empty() && (row.back() == '\n' || row.back() == '\r')) row.pop_back();

        std::string fields[4];
        std::size_t begin = 0;
        int field = 0;
        while (field < 4) {
            const std::size_t comma = row.find(',', begin);
            fields[field++] = row.substr(begin, comma == std::string::npos ? std::string::npos : comma - begin);
            if (comma == std::string::npos) break;
            begin = comma + 1;
        }
        if (field < 4) continue;

        try {
            const int id = std::stoi(fields[0]);
            GpuTelemetry telemetry;
            try { telemetry.watts = std::stod(fields[1]); telemetry.power_available = true; } catch (...) {}
            try { telemetry.temperature_c = std::stod(fields[2]); telemetry.temperature_available = true; } catch (...) {}
            try { telemetry.fan_percent = std::stod(fields[3]); telemetry.fan_available = true; } catch (...) {}
            result.devices[id] = telemetry;
            result.available = true;
        } catch (...) {}
    }
#ifdef _WIN32
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    return result;
}

struct CpuTelemetry {
    double watts{0.0};
    double temperature_c{0.0};
    bool power_available{false};
    bool temperature_available{false};
};

struct RaplSample {
    std::uint64_t energy_uj{0};
    std::uint64_t max_energy_uj{0};
    std::chrono::steady_clock::time_point time{};
    std::string energy_path;
    bool valid{false};
};

inline bool read_text_file(const std::filesystem::path& path, std::string& value)
{
    std::ifstream file(path);
    if (!file) return false;
    std::getline(file, value);
    return !value.empty();
}

inline bool read_u64_file(const std::filesystem::path& path, std::uint64_t& value)
{
    std::ifstream file(path);
    if (!file) return false;
    file >> value;
    return !file.fail();
}

inline std::string find_rapl_energy_path()
{
#ifndef _WIN32
    const std::filesystem::path root("/sys/class/powercap");
    std::error_code ec;
    if (!std::filesystem::exists(root, ec)) return {};
    for (const auto& entry : std::filesystem::directory_iterator(root, ec)) {
        if (ec || !entry.is_directory()) continue;
        std::string name;
        read_text_file(entry.path() / "name", name);
        if (name.find("package") == std::string::npos && name.find("Package") == std::string::npos) continue;
        const auto energy = entry.path() / "energy_uj";
        if (std::filesystem::exists(energy, ec)) return energy.string();
    }
#endif
    return {};
}

inline RaplSample read_rapl_sample(const std::string& existing_path = {})
{
    RaplSample sample;
#ifndef _WIN32
    sample.energy_path = existing_path.empty() ? find_rapl_energy_path() : existing_path;
    if (sample.energy_path.empty()) return sample;
    if (!read_u64_file(sample.energy_path, sample.energy_uj)) return sample;
    const std::filesystem::path base = std::filesystem::path(sample.energy_path).parent_path();
    read_u64_file(base / "max_energy_range_uj", sample.max_energy_uj);
    sample.time = std::chrono::steady_clock::now();
    sample.valid = true;
#endif
    return sample;
}

inline double query_cpu_temperature_c(bool& available)
{
    available = false;
#ifndef _WIN32
    const std::filesystem::path root("/sys/class/hwmon");
    std::error_code ec;
    if (!std::filesystem::exists(root, ec)) return 0.0;
    double best = 0.0;
    int best_priority = -1;
    for (const auto& entry : std::filesystem::directory_iterator(root, ec)) {
        if (ec || !entry.is_directory()) continue;
        std::string name;
        if (!read_text_file(entry.path() / "name", name)) continue;
        int priority = 0;
        if (name.find("k10temp") != std::string::npos || name.find("zenpower") != std::string::npos) priority = 3;
        else if (name.find("coretemp") != std::string::npos) priority = 3;
        else if (name.find("cpu") != std::string::npos) priority = 1;
        if (priority == 0) continue;
        for (int i = 1; i <= 32; ++i) {
            std::uint64_t millideg = 0;
            if (!read_u64_file(entry.path() / ("temp" + std::to_string(i) + "_input"), millideg)) continue;
            const double temp = static_cast<double>(millideg) / 1000.0;
            if (temp < 0.0 || temp > 130.0) continue;
            if (priority > best_priority || (priority == best_priority && temp > best)) {
                best_priority = priority;
                best = temp;
                available = true;
            }
        }
    }
    return best;
#else
    return 0.0;
#endif
}

inline RaplSample& previous_rapl_sample()
{
    static RaplSample sample;
    return sample;
}

inline void prime_cpu_power_sample()
{
#ifndef _WIN32
    previous_rapl_sample() = read_rapl_sample();
#endif
}

inline CpuTelemetry query_cpu_telemetry()
{
    CpuTelemetry telemetry;
    telemetry.temperature_c = query_cpu_temperature_c(telemetry.temperature_available);
#ifndef _WIN32
    RaplSample current = read_rapl_sample(previous_rapl_sample().energy_path);
    RaplSample& previous = previous_rapl_sample();
    if (current.valid && previous.valid) {
        const double seconds = std::chrono::duration<double>(current.time - previous.time).count();
        if (seconds > 0.0) {
            std::uint64_t delta = 0;
            if (current.energy_uj >= previous.energy_uj) delta = current.energy_uj - previous.energy_uj;
            else if (current.max_energy_uj > previous.energy_uj)
                delta = (current.max_energy_uj - previous.energy_uj) + current.energy_uj;
            if (delta > 0) {
                telemetry.watts = (static_cast<double>(delta) / 1000000.0) / seconds;
                telemetry.power_available = true;
            }
        }
    }
    if (current.valid) previous = current;
#endif
    return telemetry;
}

inline std::string format_power(const GpuTelemetry& telemetry)
{
    if (!telemetry.power_available) return "n/a";
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1) << telemetry.watts << " W";
    return ss.str();
}

inline std::string format_power(const CpuTelemetry& telemetry)
{
    if (!telemetry.power_available) return "n/a";
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(1) << telemetry.watts << " W";
    return ss.str();
}

inline std::string format_temperature(const GpuTelemetry& telemetry)
{
    if (!telemetry.temperature_available) return "n/a";
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(0) << telemetry.temperature_c << " C";
    return ss.str();
}

inline std::string format_temperature(const CpuTelemetry& telemetry)
{
    if (!telemetry.temperature_available) return "n/a";
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(0) << telemetry.temperature_c << " C";
    return ss.str();
}

inline std::string format_fan(const GpuTelemetry& telemetry)
{
    if (!telemetry.fan_available) return "n/a";
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(0) << telemetry.fan_percent << '%';
    return ss.str();
}

inline double status_row_hashrate_hps(const std::string& line)
{
    std::istringstream ss(line);
    std::string source;
    double value = 0.0;
    std::string unit;
    if (!(ss >> source)) return 0.0;
    if (source == "GPU") {
        int id = 0;
        if (!(ss >> id)) return 0.0;
    }
    if (!(ss >> value >> unit)) return 0.0;
    if (unit == "MH/s") return value * 1000000.0;
    if (unit == "kH/s") return value * 1000.0;
    return unit == "H/s" ? value : 0.0;
}

inline std::string format_efficiency(double hps, const GpuTelemetry& telemetry)
{
    if (!telemetry.power_available || telemetry.watts <= 0.0 || hps <= 0.0) return "n/a";
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << hps / telemetry.watts << " H/W";
    return ss.str();
}

inline std::string format_efficiency(double hps, const CpuTelemetry& telemetry)
{
    if (!telemetry.power_available || telemetry.watts <= 0.0 || hps <= 0.0) return "n/a";
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << hps / telemetry.watts << " H/W";
    return ss.str();
}

class LineColorBuf final : public std::streambuf {
public:
    LineColorBuf(std::streambuf* destination, bool enabled) : destination_(destination), enabled_(enabled) {}

protected:
    int overflow(int ch) override
    {
        if (ch == traits_type::eof()) return traits_type::not_eof(ch);
        auto& pending = thread_pending()[this];
        const char c = static_cast<char>(ch);
        if (c == '\n') emit_line(pending, true); else pending.push_back(c);
        return ch;
    }

    std::streamsize xsputn(const char* s, std::streamsize count) override
    {
        auto& pending = thread_pending()[this];
        for (std::streamsize i = 0; i < count; ++i) {
            if (s[i] == '\n') emit_line(pending, true); else pending.push_back(s[i]);
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
    static std::mutex& output_mutex() { static std::mutex mutex; return mutex; }
    static std::string& rotation_fingerprint() { static std::string value; return value; }
    static std::string& rotation_cn_variants() { static std::string value; return value; }
    static GpuTelemetrySnapshot& status_telemetry() { static GpuTelemetrySnapshot value; return value; }
    static CpuTelemetry& status_cpu_telemetry() { static CpuTelemetry value; return value; }
    static bool& in_status_table() { static bool value = false; return value; }

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

    static int gpu_id_from_status_row(const std::string& line)
    {
        const std::size_t pos = line.find("GPU ");
        if (pos == std::string::npos) return -1;
        try { return std::stoi(line.substr(pos + 4)); } catch (...) { return -1; }
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

        const bool top = pending.find("PROOF OF GRASS | STATUS UPDATE") != std::string::npos;
        if (top) {
            status_telemetry() = query_gpu_telemetry();
            status_cpu_telemetry() = query_cpu_telemetry();
            in_status_table() = true;
        }

        std::string rendered = pending;
        if (in_status_table() && rendered.find("SOURCE") != std::string::npos && rendered.find("HASHRATE") != std::string::npos) {
            rendered += "POWER       TEMP     FAN      EFFICIENCY";
        } else if (in_status_table() && rendered.find("CPU") != std::string::npos && rendered.find("H/s") != std::string::npos) {
            const double hps = status_row_hashrate_hps(rendered);
            rendered += "        " + format_power(status_cpu_telemetry())
                      + "    " + format_temperature(status_cpu_telemetry())
                      + "    -        " + format_efficiency(hps, status_cpu_telemetry());
        } else if (in_status_table() && rendered.find("GPU ") != std::string::npos && rendered.find("H/s") != std::string::npos) {
            const int id = gpu_id_from_status_row(rendered);
            const auto it = status_telemetry().devices.find(id);
            if (it != status_telemetry().devices.end()) {
                const double hps = status_row_hashrate_hps(rendered);
                rendered += "        " + format_power(it->second)
                          + "    " + format_temperature(it->second)
                          + "    " + format_fan(it->second)
                          + "    " + format_efficiency(hps, it->second);
            } else {
                rendered += "        n/a         n/a      n/a      n/a";
            }
        }

        const char* color = enabled_ ? color_for_line(pending) : nullptr;
        if (color != nullptr) destination_->sputn(color, static_cast<std::streamsize>(std::char_traits<char>::length(color)));
        if (!rendered.empty()) destination_->sputn(rendered.data(), static_cast<std::streamsize>(rendered.size()));
        if (color != nullptr) destination_->sputn(kReset, 4);
        if (newline) destination_->sputc('\n');

        if (newline && top) {
            if (!rotation_fingerprint().empty()) write_raw_line("ROTATION      " + rotation_fingerprint());
            if (!rotation_cn_variants().empty()) write_raw_line("CRYPTONIGHT   " + rotation_cn_variants());
        }
        if (pending.find("========================================================================================================") != std::string::npos && !top)
            in_status_table() = false;
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
    detail::prime_cpu_power_sample();
    const bool enabled = detail::terminal_supports_color();
    static detail::LineColorBuf cout_buffer(std::cout.rdbuf(), enabled);
    static detail::LineColorBuf cerr_buffer(std::cerr.rdbuf(), enabled);
    std::cout.rdbuf(&cout_buffer);
    std::cerr.rdbuf(&cerr_buffer);
}
} // namespace yerbas::console
