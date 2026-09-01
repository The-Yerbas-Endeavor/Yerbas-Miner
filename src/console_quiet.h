#pragma once

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <streambuf>
#include <string>
#include <vector>

namespace yerbas::console {
namespace quiet_detail {

inline std::string& perf_csv_path()
{
    static std::string path;
    return path;
}

inline std::mutex& perf_csv_mutex()
{
    static std::mutex mutex;
    return mutex;
}

inline std::string& current_rotation()
{
    static std::string value;
    return value;
}

inline std::string& current_cn()
{
    static std::string value;
    return value;
}

inline std::string csv_escape(const std::string& value)
{
    std::string out = "\"";
    for (char c : value) {
        if (c == '"') out += "\"\"";
        else out += c;
    }
    out += '"';
    return out;
}

inline std::string strip_ansi(const std::string& line)
{
    std::string clean;
    clean.reserve(line.size());
    for (std::size_t i = 0; i < line.size();) {
        if (line[i] == '\x1b' && i + 1 < line.size() && line[i + 1] == '[') {
            i += 2;
            while (i < line.size()) {
                const unsigned char c = static_cast<unsigned char>(line[i++]);
                if (c >= 0x40 && c <= 0x7e) break;
            }
            continue;
        }
        clean.push_back(line[i++]);
    }
    return clean;
}

inline std::string field_after(const std::string& line, const std::string& key, const std::string& end = " | ")
{
    const auto pos = line.find(key);
    if (pos == std::string::npos) return {};
    const auto start = pos + key.size();
    const auto stop = line.find(end, start);
    return line.substr(start, stop == std::string::npos ? std::string::npos : stop - start);
}

inline bool ensure_perf_csv_ready(const std::string& path)
{
    if (path.empty()) return true;
    try {
        const std::filesystem::path target(path);
        if (target.has_parent_path())
            std::filesystem::create_directories(target.parent_path());

        std::ifstream check(path, std::ios::binary);
        const bool needs_header = !check.good() || check.peek() == std::ifstream::traits_type::eof();
        check.close();

        std::ofstream out(path, std::ios::app);
        if (!out) return false;
        if (needs_header) {
            out << "timestamp_ms,event,rotation,cryptonight,source,hps,total_ms,stage,stage_ms,stage_pct,details\n";
            out.flush();
        }
        return out.good();
    } catch (...) {
        return false;
    }
}

inline void append_perf_row(const std::string& event,
                            const std::string& rotation,
                            const std::string& cn,
                            const std::string& source,
                            const std::string& hps,
                            const std::string& total_ms,
                            const std::string& stage,
                            const std::string& stage_ms,
                            const std::string& stage_pct,
                            const std::string& details)
{
    const auto& path = perf_csv_path();
    if (path.empty()) return;
    std::lock_guard<std::mutex> lock(perf_csv_mutex());
    if (!ensure_perf_csv_ready(path)) return;
    std::ofstream out(path, std::ios::app);
    if (!out) return;
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    out << now_ms << ','
        << csv_escape(event) << ','
        << csv_escape(rotation) << ','
        << csv_escape(cn) << ','
        << csv_escape(source) << ','
        << csv_escape(hps) << ','
        << csv_escape(total_ms) << ','
        << csv_escape(stage) << ','
        << csv_escape(stage_ms) << ','
        << csv_escape(stage_pct) << ','
        << csv_escape(details) << '\n';
    out.flush();
}

inline void capture_status_row(const std::string& raw)
{
    const std::string line = strip_ansi(raw);
    std::istringstream ss(line);
    std::string first;
    if (!(ss >> first)) return;

    std::string source;
    if (first == "CPU") {
        source = "CPU";
    } else if (first == "GPU") {
        std::string id;
        if (!(ss >> id)) return;
        source = "GPU " + id;
    } else if (first == "TOTAL") {
        source = "TOTAL";
    } else {
        return;
    }

    double value = 0.0;
    std::string unit;
    if (!(ss >> value >> unit)) return;
    if (unit == "kH/s") value *= 1000.0;
    else if (unit == "MH/s") value *= 1000000.0;
    else if (unit != "H/s") return;

    std::ostringstream hps;
    hps << value;
    append_perf_row("status", current_rotation(), current_cn(), source, hps.str(), "", "", "", "", line);
}

inline void capture_perf_line(const std::string& raw_line)
{
    if (perf_csv_path().empty() || raw_line.empty()) return;
    const std::string line = strip_ansi(raw_line);

    if (line.find("[GhostRider] rotation=") != std::string::npos) {
        current_rotation() = field_after(line, "rotation=", " | ");
        current_cn() = field_after(line, "CN=", "\n");
        append_perf_row("rotation", current_rotation(), current_cn(), "", "", "", "", "", "", line);
        return;
    }

    if (line.find("[CPU fingerprint]") != std::string::npos) {
        std::string hps = field_after(line, "live=", " H/s");
        append_perf_row("cpu_fingerprint", field_after(line, "rotation=", " | "), field_after(line, "CN=", " | "), "CPU", hps, "", "", "", "", line);
        return;
    }

    if (line.find("[CUDA GR perf] GPU ") != std::string::npos) {
        const auto gpu_pos = line.find("GPU ");
        const auto gpu_end = line.find(" |", gpu_pos);
        const std::string source = gpu_pos == std::string::npos ? "GPU" : line.substr(gpu_pos, gpu_end - gpu_pos);
        append_perf_row("cuda_gr", field_after(line, "fingerprint=", " | "), field_after(line, "CN=", " | "), source,
                        field_after(line, "throughput=", " H/s"), field_after(line, "stages=", " ms"), "", "", "", line);
        return;
    }

    if (line.find("[CUDA CN stagger tuner] GPU ") != std::string::npos) {
        const auto gpu_pos = line.find("GPU ");
        const auto gpu_end = line.find(" |", gpu_pos);
        const std::string source = gpu_pos == std::string::npos ? "GPU" : line.substr(gpu_pos, gpu_end - gpu_pos);
        std::string cn;
        if (gpu_end != std::string::npos) {
            const auto cn_start = gpu_end + 3U;
            const auto cn_end = line.find(" |", cn_start);
            if (cn_end != std::string::npos) cn = line.substr(cn_start, cn_end - cn_start);
        }
        append_perf_row("cuda_cn_stagger_tuner", current_rotation(), cn, source, "",
                        field_after(line, "single=", " ms"),
                        field_after(line, "production=", " | "),
                        field_after(line, "stagger33=", " ms"),
                        field_after(line, "stagger50=", " ms"), line);
        return;
    }

    if (line.find("[CUDA CN stagger] GPU ") != std::string::npos) {
        const auto gpu_pos = line.find("GPU ");
        const auto gpu_end = line.find(" |", gpu_pos);
        const std::string source = gpu_pos == std::string::npos ? "GPU" : line.substr(gpu_pos, gpu_end - gpu_pos);
        std::string cn;
        if (gpu_end != std::string::npos) {
            const auto cn_start = gpu_end + 3U;
            const auto cn_end = line.find(" |", cn_start);
            if (cn_end != std::string::npos) cn = line.substr(cn_start, cn_end - cn_start);
        }
        append_perf_row("cuda_cn_stagger", current_rotation(), cn, source, "", "",
                        field_after(line, "mode=", " | "), "", "", line);
        return;
    }

    if (line.find("[CUDA CN phase] GPU ") != std::string::npos) {
        const auto gpu_pos = line.find("GPU ");
        const auto gpu_end = line.find(" |", gpu_pos);
        const std::string source = gpu_pos == std::string::npos ? "GPU" : line.substr(gpu_pos, gpu_end - gpu_pos);

        std::string cn;
        if (gpu_end != std::string::npos) {
            const auto cn_start = gpu_end + 3U;
            const auto cn_end = line.find(" |", cn_start);
            if (cn_end != std::string::npos) cn = line.substr(cn_start, cn_end - cn_start);
        }

        const std::string total_ms = field_after(line, "total=", " ms");
        append_perf_row("cuda_cn_phase", current_rotation(), cn, source, "", total_ms, "setup",
                        field_after(line, "setup=", " ms"), "", line);
        append_perf_row("cuda_cn_phase", current_rotation(), cn, source, "", total_ms, "loop",
                        field_after(line, "loop=", " ms"), "", line);
        append_perf_row("cuda_cn_phase", current_rotation(), cn, source, "", total_ms, "final",
                        field_after(line, "final=", " ms"), "", line);
        return;
    }

    if (line.find("[CPU stage profile]") != std::string::npos) {
        append_perf_row("cpu_stage_profile", field_after(line, "rotation=", " | "), "", "CPU", "",
                        field_after(line, "total=", " ms/hash"), field_after(line, "hot=", " "), "", field_after(line, "(", "%)"), line);
        return;
    }

    if (line.find("[CUDA stage]") != std::string::npos) {
        const auto marker = line.find("[CUDA stage]");
        const auto pipe = line.find(" |", marker);
        const std::string stage = marker == std::string::npos || pipe == std::string::npos ? "" : line.substr(marker + 12, pipe - (marker + 12));
        append_perf_row("cuda_stage", "", "", "GPU", "", "", stage,
                        field_after(line, "sample=", " ms"), field_after(line, "sample=", "%"), line);
        return;
    }

    capture_status_row(line);
}

inline bool diagnostics_enabled()
{
    const char* value = std::getenv("YERBAS_DIAGNOSTICS");
    return value != nullptr && *value != '\0' && std::string(value) != "0";
}

inline bool suppress_normal_line(const std::string& line)
{
    if (line.find("[CPU GR perf]") != std::string::npos) return true;
    if (line.find("[CUDA GR perf]") != std::string::npos) return true;
    if (line.find("[CUDA stage]") != std::string::npos) return true;
    if (line.find("GhostRider rolling stage profile") != std::string::npos) return true;
    if (line.find("[CUDA CN runtime]") != std::string::npos) return true;
    if (line.find("[CUDA CN phase]") != std::string::npos) return true;
    if (line.find("] candidate | job=") != std::string::npos) return true;
    if (line.find("[CUDA CN profile]") != std::string::npos) return true;
    if (line.find("[CPU CN profile]") != std::string::npos) return true;
    if (line.find("[CPU CN phase]") != std::string::npos) return true;
    if (line.find("[CPU CN candidate]") != std::string::npos) return true;
    if (line.find("[CPU CN A/B]") != std::string::npos) return true;
    if (line.find("[CUDA CN production test]") != std::string::npos) return true;
    if (line.find("[CUDA CN coop test]") != std::string::npos) return true;
    if (line.find("[CUDA dual-state test]") != std::string::npos) return true;
    if (line.find("CryptoNight geometry cache loaded |") != std::string::npos) return true;
    if (line.find("CryptoNight loop cache loaded |") != std::string::npos) return true;
    if (line.find("CryptoNight loop AES cache loaded |") != std::string::npos) return true;
    if (line.find("CryptoNight dual-state cache loaded |") != std::string::npos) return true;
    if (line.find("CryptoNight production loop cache loaded |") != std::string::npos) return true;
    if (line.find("CryptoNight cooperative cache loaded |") != std::string::npos) return true;
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
    static bool is_status_top(const std::string& line)
    {
        return strip_ansi(line).find("PROOF OF GRASS | STATUS UPDATE") != std::string::npos;
    }

    static bool is_status_bottom(const std::string& line)
    {
        const std::string clean = strip_ansi(line);
        return clean.find("========================================================================================================") != std::string::npos &&
               clean.find("PROOF OF GRASS | STATUS UPDATE") == std::string::npos;
    }

    void forward_line(const std::string& line, bool newline)
    {
        capture_perf_line(line);
        const bool suppress = !diagnostics_enabled() && suppress_normal_line(line);
        if (!suppress && !line.empty())
            destination_->sputn(line.data(), static_cast<std::streamsize>(line.size()));
        if (!suppress && newline) destination_->sputc('\n');
    }

    void flush_status_buffer()
    {
        bool empty_snapshot = false;
        for (const auto& line : status_lines_) {
            const std::string clean = strip_ansi(line);
            if (clean.rfind("TOTAL", 0) == 0 && clean.find("0.00 H/s") != std::string::npos) {
                empty_snapshot = true;
                break;
            }
        }

        if (!empty_snapshot) {
            for (const auto& line : status_lines_) forward_line(line, true);
        }
        status_lines_.clear();
        buffering_status_ = false;
    }

    void emit(bool newline)
    {
        if (buffering_status_) {
            status_lines_.push_back(pending_);
            const bool bottom = is_status_bottom(pending_);
            pending_.clear();
            if (bottom) flush_status_buffer();
            return;
        }

        if (newline && is_status_top(pending_)) {
            buffering_status_ = true;
            status_lines_.push_back(pending_);
            pending_.clear();
            return;
        }

        forward_line(pending_, newline);
        pending_.clear();
    }

    std::streambuf* destination_{nullptr};
    std::string pending_;
    bool buffering_status_{false};
    std::vector<std::string> status_lines_;
};

} // namespace quiet_detail

inline void set_perf_csv_path(const std::string& path)
{
    quiet_detail::perf_csv_path() = path;
    if (!quiet_detail::ensure_perf_csv_ready(path)) {
        std::cerr << "Performance CSV error: unable to create/write " << path << '\n';
    }
}

inline void enable_quiet_output()
{
    static bool initialized = false;
    if (initialized) return;
    initialized = true;
    static quiet_detail::QuietLineBuf quiet_cout(std::cout.rdbuf());
    std::cout.rdbuf(&quiet_cout);
}

} // namespace yerbas::console
