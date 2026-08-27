#include "config.h"
#include "console.h"
#include "console_quiet.h"
#include "first_run.h"
#include "miner.h"

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <streambuf>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

void write_startup_log(const std::string& message)
{
    std::ofstream log("yerbas-miner-startup.log", std::ios::app);
    if (log) log << message << '\n';
}

struct StreamBufferRestore {
    std::streambuf* cout_buf{nullptr};
    std::streambuf* cerr_buf{nullptr};

    StreamBufferRestore() noexcept
        : cout_buf(std::cout.rdbuf()), cerr_buf(std::cerr.rdbuf())
    {
    }

    ~StreamBufferRestore() noexcept
    {
        try { std::cout.flush(); } catch (...) {}
        try { std::cerr.flush(); } catch (...) {}
        if (cout_buf != nullptr) std::cout.rdbuf(cout_buf);
        if (cerr_buf != nullptr) std::cerr.rdbuf(cerr_buf);
    }
};

#ifdef _WIN32
const char* windows_access_kind(ULONG_PTR kind)
{
    switch (kind) {
    case 0: return "read";
    case 1: return "write";
    case 8: return "execute";
    default: return "unknown";
    }
}

LONG WINAPI windows_unhandled_exception_filter(EXCEPTION_POINTERS* info)
{
    std::ostringstream ss;
    ss << "Fatal Windows exception";

    if (info != nullptr && info->ExceptionRecord != nullptr) {
        const EXCEPTION_RECORD* record = info->ExceptionRecord;
        const auto exception_address = reinterpret_cast<std::uintptr_t>(record->ExceptionAddress);

        ss << " | code=0x" << std::hex << std::uppercase
           << static_cast<unsigned long>(record->ExceptionCode)
           << " | address=0x" << exception_address;

        if (record->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && record->NumberParameters >= 2) {
            ss << " | access=" << windows_access_kind(record->ExceptionInformation[0])
               << " | fault_address=0x" << static_cast<std::uintptr_t>(record->ExceptionInformation[1]);
        }

        MEMORY_BASIC_INFORMATION mbi{};
        if (VirtualQuery(record->ExceptionAddress, &mbi, sizeof(mbi)) == sizeof(mbi) && mbi.AllocationBase != nullptr) {
            const auto module_base = reinterpret_cast<std::uintptr_t>(mbi.AllocationBase);
            ss << " | module_base=0x" << module_base
               << " | module_offset=0x" << (exception_address - module_base);

            char module_path[MAX_PATH]{};
            const DWORD path_len = GetModuleFileNameA(static_cast<HMODULE>(mbi.AllocationBase), module_path, static_cast<DWORD>(sizeof(module_path)));
            if (path_len > 0 && path_len < sizeof(module_path)) ss << " | module=" << module_path;
        }
    }

    const std::string message = ss.str();
    write_startup_log(message);
    std::ofstream crash("yerbas-miner-crash.log", std::ios::app);
    if (crash) crash << message << '\n';
    return EXCEPTION_EXECUTE_HANDLER;
}

void pause_on_windows_error()
{
    std::cerr << "\nPress Enter to close..." << std::flush;
    std::cin.get();
}
#endif

} // namespace

int main(int argc, char** argv)
{
#ifdef _WIN32
    SetUnhandledExceptionFilter(windows_unhandled_exception_filter);
#endif

    StreamBufferRestore restore_streams;
    yerbas::console::enable_colors();
    yerbas::console::enable_quiet_output();
    write_startup_log("Yerbas Miner starting");

    std::cout << "\nYerbas Miner starting...\n"
              << "------------------------------------------------------------\n"
              << std::flush;

    try {
        auto config = yerbas::load_config(argc, argv);
        yerbas::first_run::apply(config);
        if (!config.logging.perf_csv.empty()) {
            yerbas::console::set_perf_csv_path(config.logging.perf_csv);
            std::cout << "Performance CSV: " << config.logging.perf_csv << '\n';
        }
        yerbas::Miner miner(config);
        const int result = miner.run();
        write_startup_log("Yerbas Miner exited with code " + std::to_string(result));
        return result;
    } catch (const std::exception& e) {
        const std::string message = std::string("Fatal: ") + e.what();
        write_startup_log(message);
        std::cerr << message << '\n';
#ifdef _WIN32
        pause_on_windows_error();
#endif
        return 1;
    } catch (...) {
        write_startup_log("Fatal: unknown startup exception");
        std::cerr << "Fatal: unknown startup exception\n";
#ifdef _WIN32
        pause_on_windows_error();
#endif
        return 1;
    }
}
