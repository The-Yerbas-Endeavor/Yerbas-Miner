#include "config.h"
#include "miner.h"

#include <exception>
#include <iostream>

int main(int argc, char** argv)
{
    try {
        const auto config = yerbas::load_config(argc, argv);
        yerbas::Miner miner(config);
        return miner.run();
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << '\n';
        return 1;
    }
}
