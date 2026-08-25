#pragma once

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <streambuf>
#include <string>

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

inline std::string field_after(const std::string& line, const std::string& key, const std::string& end = " | ")
{
    const auto pos = line.find(key);
    if (pos == std::string::npos) return {};
    const auto start = pos + key.size();
    const auto stop = line.find(end, start);
    return line.substr(start, stop == std::string::npos ? std::string::npos : stop - start);
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
    std::ifstream check(path);
    const bool needs_header = !check.good() || check.peek() == std::ifstream::traits_type::eof();
    check.close();
    std::ofstream out(path, std::ios::app);
    if (!out) return;
    if (needs_header)
        out << "timestamp_ms,event,rotation,cryptonight,source,hps,total_ms,stage,stage_ms,stage_pct,details\n";
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

inline void capture_perf_line(const std::string& line)
{
    if (perf_csv_path().empty() || line.empty()) return;

    if (line.find("[GhostRider] rotation=") != std::string::npos) {
        append_perf_row("rotation", field_after(line, "rotation=", " | "), field_after(line, "CN=", "\n"), "", "", "", "", "", "", line);
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
    void emit(bool newline)
    {
        capture_perf_line(pending_);
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

inline void set_perf_csv_path(const std::string& path)
{
    quiet_detail::perf_csv_path() = path;
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
