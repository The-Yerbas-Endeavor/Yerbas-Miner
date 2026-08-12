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

# Header-only JSON parser used by config.json. It is linked into the binary and
# does not add a runtime DLL/shared-library dependency.
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

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

# Yerbas Core's CryptoNight code uses _malloca() under MSVC but releases that
# buffer with free(). Patch only the fetched build-tree copy so the pinned
# upstream repository remains unchanged.
if(WIN32)
    set(YERBAS_SLOW_HASH "${YERBAS_CORE_SRC}/cryptonote/slow-hash.c")
    file(READ "${YERBAS_SLOW_HASH}" YERBAS_SLOW_HASH_CONTENT)
    string(FIND "${YERBAS_SLOW_HASH_CONTENT}" "  free(long_state);" YERBAS_FREE_POS)
    if(YERBAS_FREE_POS EQUAL -1)
        message(FATAL_ERROR "Could not locate CryptoNight long_state cleanup in pinned slow-hash.c")
    endif()
    string(REPLACE
        "  free(long_state);"
        "#if defined(_MSC_VER)\n  _freea(long_state);\n#else\n  free(long_state);\n#endif"
        YERBAS_SLOW_HASH_CONTENT
        "${YERBAS_SLOW_HASH_CONTENT}"
    )
    file(WRITE "${YERBAS_SLOW_HASH}" "${YERBAS_SLOW_HASH_CONTENT}")
endif()

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

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_options(yerbas-ghostrider-reference PRIVATE
        -include "${CMAKE_CURRENT_SOURCE_DIR}/cmake/glibc_endian_compat.h"
    )
endif()

if(WIN32)
    target_include_directories(yerbas-ghostrider-reference BEFORE PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/compat/windows"
    )
    target_sources(yerbas-ghostrider-reference PRIVATE
        compat/windows/gettimeofday.c
    )
endif()

add_executable(yerbas-miner
    src/main.cpp
    src/miner.cpp
    src/config.cpp
    src/stratum/stratum.cpp
)
target_include_directories(yerbas-miner PRIVATE src)
target_link_libraries(yerbas-miner PRIVATE
    yerbas-ghostrider-reference
    nlohmann_json::nlohmann_json
)

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
