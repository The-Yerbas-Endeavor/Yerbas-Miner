#include "config.h"
#include "console.h"
#include "miner.h"

#include <exception>
#include <fstream>
#include <iostream>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

// CUDA 12.x commonly uses an older supported GCC host compiler while the rest
// of Yerbas-Miner is built with the system GCC (for example GCC 15 on Ubuntu
// 26). GCC 14+ may emit __cxa_call_terminate in noexcept cleanup paths, while
// an older libstdc++ selected through nvcc's GCC library path may not export
// that ABI helper. Its required behavior is simply to terminate while handling
// an exception, so provide a weak compatibility bridge for mixed GNU toolchains.
#if defined(__linux__) && defined(__GNUC__) && (__GNUC__ >= 14)
extern "C" [[noreturn]] __attribute__((weak)) void __cxa_call_terminate(void*) noexcept
{
    std::terminate();
}
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
void pause_on_windows_error()
{
    std::cerr << "\nPress Enter to close..." << std::flush;
    std::cin.get();
}
#endif

} // namespace

int main(int argc, char** argv)
{
    yerbas::console::enable_colors();
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
