#include "miner.h"

#include <exception>
#include <iostream>

int main()
{
    try {
        yerbas::Miner miner;
        return miner.run();
    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << '\n';
        return 1;
    }
}
