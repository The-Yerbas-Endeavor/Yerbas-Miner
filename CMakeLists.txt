cmake_minimum_required(VERSION 3.24)
project(YerbasMiner VERSION 0.5.2 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Release mining builds benefit materially from whole-program optimization,
# especially because the GhostRider reference path crosses C/C++ translation
# unit boundaries on every hash.
include(CheckIPOSupported)
check_ipo_supported(RESULT YERBAS_IPO_SUPPORTED OUTPUT YERBAS_IPO_ERROR)
if(YERBAS_IPO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

option(YERBAS_ENABLE_CUDA "Build the NVIDIA CUDA backend" ON)
option(YERBAS_FETCH_CORE "Fetch the pinned Yerbas Core GhostRider reference sources" ON)

if(NOT YERBAS_FETCH_CORE)
    message(FATAL_ERROR "Yerbas Core GhostRider sources are required for the CPU reference implementation")
endif()

include(FetchContent)
include(CTest)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

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
target_include_directories(yerbas-ghostrider-reference PUBLIC src PRIVATE ${YERBAS_CORE_SRC})

if(MSVC)
    target_compile_options(yerbas-ghostrider-reference PRIVATE $<$<CONFIG:Release>:/O2 /Ob3 /Oi /Ot>)
else()
    target_compile_options(yerbas-ghostrider-reference PRIVATE $<$<CONFIG:Release>:-O3>)
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_options(yerbas-ghostrider-reference PRIVATE
        -include "${CMAKE_CURRENT_SOURCE_DIR}/cmake/glibc_endian_compat.h"
    )
endif()

if(WIN32)
    target_include_directories(yerbas-ghostrider-reference BEFORE PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/compat/windows"
    )
    target_sources(yerbas-ghostrider-reference PRIVATE compat/windows/gettimeofday.c)
endif()

add_executable(yerbas-miner
    src/main.cpp
    src/miner.cpp
    src/config.cpp
    src/crypto/sha256.cpp
    src/stratum/stratum.cpp
)
target_include_directories(yerbas-miner PRIVATE src)
target_link_libraries(yerbas-miner PRIVATE yerbas-ghostrider-reference nlohmann_json::nlohmann_json)

if(MSVC)
    target_compile_options(yerbas-miner PRIVATE $<$<CONFIG:Release>:/O2 /Ob3 /Oi /Ot>)
else()
    target_compile_options(yerbas-miner PRIVATE $<$<CONFIG:Release>:-O3>)
endif()

if(WIN32)
    target_link_libraries(yerbas-miner PRIVATE ws2_32)
endif()

set(YERBAS_CUDA_AVAILABLE OFF)
if(YERBAS_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR
            "YERBAS_ENABLE_CUDA=ON but no CUDA compiler was detected. "
            "Refusing to produce a CPU-only binary labeled as a CUDA build.")
    endif()

    enable_language(CUDA)
    set(YERBAS_CUDA_AVAILABLE ON)

    set(YERBAS_CUDA_ARCHITECTURES "52;60;61;70;75;80;86;89;90" CACHE STRING
        "CUDA compute capabilities built into Yerbas-Miner")
    message(STATUS "Yerbas CUDA backend enabled with compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "Yerbas CUDA architectures: ${YERBAS_CUDA_ARCHITECTURES}")

    target_sources(yerbas-miner PRIVATE
        src/cuda/cuda_backend.cu
        src/cuda/blake_validation.cu
        src/cuda/skein_validation.cu
    )
    target_compile_definitions(yerbas-miner PRIVATE YERBAS_HAS_CUDA=1)
    set_target_properties(yerbas-miner PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${YERBAS_CUDA_ARCHITECTURES}"
    )
endif()

if(BUILD_TESTING)
    add_executable(ghostrider-smoke tests/ghostrider_smoke.cpp)
    target_include_directories(ghostrider-smoke PRIVATE src)
    target_link_libraries(ghostrider-smoke PRIVATE yerbas-ghostrider-reference)
    add_test(NAME ghostrider-smoke COMMAND ghostrider-smoke)

    if(YERBAS_CUDA_AVAILABLE)
        add_executable(cuda-keccak-validation
            tests/cuda_keccak_validation.cpp
            src/cuda/cuda_backend.cu
            src/cuda/blake_validation.cu
            src/cuda/skein_validation.cu
        )
        target_include_directories(cuda-keccak-validation PRIVATE src tests)
        target_link_libraries(cuda-keccak-validation PRIVATE yerbas-ghostrider-reference)
        set_target_properties(cuda-keccak-validation PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "${YERBAS_CUDA_ARCHITECTURES}"
        )
        add_test(NAME cuda-keccak-validation COMMAND cuda-keccak-validation)
    endif()
endif()
