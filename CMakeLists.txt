cmake_minimum_required(VERSION 3.24)
project(YerbasMiner VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(YERBAS_ENABLE_CUDA "Build the NVIDIA CUDA backend" ON)

add_executable(yerbas-miner
    src/main.cpp
    src/miner.cpp
    src/ghostrider/ghostrider.cpp
)

target_include_directories(yerbas-miner PRIVATE src)

if(YERBAS_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        target_sources(yerbas-miner PRIVATE src/cuda/cuda_backend.cu)
        target_compile_definitions(yerbas-miner PRIVATE YERBAS_HAS_CUDA=1)
        set_target_properties(yerbas-miner PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_SEPARABLE_COMPILATION ON
        )
    else()
        message(WARNING "CUDA toolkit not found; building CPU-only scaffold")
    endif()
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(ghostrider-smoke tests/ghostrider_smoke.cpp src/ghostrider/ghostrider.cpp)
    target_include_directories(ghostrider-smoke PRIVATE src)
    add_test(NAME ghostrider-smoke COMMAND ghostrider-smoke)
endif()
