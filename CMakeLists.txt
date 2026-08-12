cmake_minimum_required(VERSION 3.24)
project(YerbasMiner VERSION 0.2.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(YERBAS_ENABLE_CUDA "Build the NVIDIA CUDA backend" ON)
option(YERBAS_FETCH_CORE "Fetch the pinned Yerbas Core GhostRider reference sources" ON)

if(NOT YERBAS_FETCH_CORE)
    message(FATAL_ERROR "Yerbas Core GhostRider sources are required for the CPU reference implementation")
endif()

include(FetchContent)

# Pin the CPU reference implementation to the exact Yerbas Core commit used as
# the source of truth for GhostRider. Do not track a moving branch: CPU/GPU hash
# equality must remain reproducible.
FetchContent_Declare(
    yerbas_core
    GIT_REPOSITORY https://github.com/The-Yerbas-Endeavor/yerbas.git
    GIT_TAG 040073f22e2b496b21e07eebfc6ca97e22b4cd40
    GIT_SHALLOW FALSE
)
FetchContent_GetProperties(yerbas_core)
if(NOT yerbas_core_POPULATED)
    FetchContent_Populate(yerbas_core)
endif()

set(YERBAS_CORE_SRC "${yerbas_core_SOURCE_DIR}/src")

# This list mirrors the GhostRider/SPHlib/CryptoNight sources used by Yerbas
# Core's crypto library and hash_selection.cpp.
set(YERBAS_GR_CORE_SOURCES
    ${YERBAS_CORE_SRC}/hash_selection.cpp

    ${YERBAS_CORE_SRC}/crypto/aes_helper.c
    ${YERBAS_CORE_SRC}/crypto/blake.c
    ${YERBAS_CORE_SRC}/crypto/bmw.c
    ${YERBAS_CORE_SRC}/crypto/cubehash.c
    ${YERBAS_CORE_SRC}/crypto/echo.c
    ${YERBAS_CORE_SRC}/crypto/groestl.c
    ${YERBAS_CORE_SRC}/crypto/jh.c
    ${YERBAS_CORE_SRC}/crypto/keccak.c
    ${YERBAS_CORE_SRC}/crypto/luffa.c
    ${YERBAS_CORE_SRC}/crypto/shavite.c
    ${YERBAS_CORE_SRC}/crypto/simd.c
    ${YERBAS_CORE_SRC}/crypto/skein.c
    ${YERBAS_CORE_SRC}/crypto/sph_hamsi.c
    ${YERBAS_CORE_SRC}/crypto/sph_hamsi_helper.c
    ${YERBAS_CORE_SRC}/crypto/sph_fugue.c
    ${YERBAS_CORE_SRC}/crypto/sph_shabal.c
    ${YERBAS_CORE_SRC}/crypto/sph_whirlpool.c
    ${YERBAS_CORE_SRC}/crypto/sph_sha2.c
    ${YERBAS_CORE_SRC}/crypto/sph_sha512.c

    ${YERBAS_CORE_SRC}/cryptonote/aesb.c
    ${YERBAS_CORE_SRC}/cryptonote/c_blake256.c
    ${YERBAS_CORE_SRC}/cryptonote/c_groestl.c
    ${YERBAS_CORE_SRC}/cryptonote/c_jh.c
    ${YERBAS_CORE_SRC}/cryptonote/c_keccak.c
    ${YERBAS_CORE_SRC}/cryptonote/c_skein.c
    ${YERBAS_CORE_SRC}/cryptonote/hash-ops.c
    ${YERBAS_CORE_SRC}/cryptonote/oaes_lib.c
    ${YERBAS_CORE_SRC}/cryptonote/slow-hash.c
)

foreach(source IN LISTS YERBAS_GR_CORE_SOURCES)
    if(NOT EXISTS "${source}")
        message(FATAL_ERROR "Pinned Yerbas Core GhostRider source is missing: ${source}")
    endif()
endforeach()

add_library(yerbas-ghostrider-reference STATIC
    src/ghostrider/ghostrider.cpp
    ${YERBAS_GR_CORE_SOURCES}
)
target_include_directories(yerbas-ghostrider-reference
    PUBLIC src
    PRIVATE ${YERBAS_CORE_SRC}
)

add_executable(yerbas-miner
    src/main.cpp
    src/miner.cpp
)
target_include_directories(yerbas-miner PRIVATE src)
target_link_libraries(yerbas-miner PRIVATE yerbas-ghostrider-reference)

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
        message(WARNING "CUDA toolkit not found; building CPU reference only")
    endif()
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(ghostrider-smoke tests/ghostrider_smoke.cpp)
    target_include_directories(ghostrider-smoke PRIVATE src)
    target_link_libraries(ghostrider-smoke PRIVATE yerbas-ghostrider-reference)
    add_test(NAME ghostrider-smoke COMMAND ghostrider-smoke)
endif()
