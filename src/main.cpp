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
