#include "config.h"
#include "console.h"
#include "console_quiet.h"
#include "miner.h"

#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

void write_startup_log(const std::string& message)
{
    std::ofstream log("yerbas-miner-startup.log", std::ios::app);
    if (log) {
        log << message << '\n';
    }
}

#ifdef _WIN32
LONG WINAPI windows_unhandled_exception_filter(EXCEPTION_POINTERS* info)
{
    std::ostringstream ss;
    ss << "Fatal Windows exception";
    if (info != nullptr && info->ExceptionRecord != nullptr) {
        ss << " | code=0x" << std::hex << std::uppercase
           << static_cast<unsigned long>(info->ExceptionRecord->ExceptionCode)
           << " | address=0x"
           << reinterpret_cast<std::uintptr_t>(info->ExceptionRecord->ExceptionAddress);
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

    yerbas::console::enable_colors();
    yerbas::console::enable_quiet_output();
    write_startup_log("Yerbas Miner starting");

    std::cout << "\nYerbas Miner starting...\n"
              << "------------------------------------------------------------\n"
              << std::flush;

    try {
        const auto config = yerbas::load_config(argc, argv);
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
