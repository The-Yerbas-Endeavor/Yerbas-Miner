cmake_minimum_required(VERSION 3.24)
project(YerbasMiner VERSION 0.5.2 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CheckIPOSupported)
check_ipo_supported(RESULT YERBAS_IPO_SUPPORTED OUTPUT YERBAS_IPO_ERROR)

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

if(YERBAS_IPO_SUPPORTED)
    set_property(TARGET yerbas-ghostrider-reference PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

if(MSVC)
    target_compile_options(yerbas-ghostrider-reference PRIVATE
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:Release>:/Oi>
        $<$<CONFIG:Release>:/Ot>
    )
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
    src/cpu/cpu_worker_pool.cpp
    src/crypto/sha256.cpp
    src/stratum/stratum.cpp
)
target_include_directories(yerbas-miner PRIVATE src)
target_link_libraries(yerbas-miner PRIVATE yerbas-ghostrider-reference nlohmann_json::nlohmann_json)

if(MSVC)
    target_compile_options(yerbas-miner PRIVATE
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C,CXX>>:/O2>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C,CXX>>:/Oi>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C,CXX>>:/Ot>
    )
else()
    target_compile_options(yerbas-miner PRIVATE
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:C,CXX>>:-O3>
    )
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

    # Local developer builds compile only for GPUs present on the machine so
    # CUDA iteration stays fast. CI/release builds produce a portable fat
    # binary spanning the supported NVIDIA generations. Either mode can be
    # overridden explicitly with -DYERBAS_CUDA_ARCHITECTURES=...
    if(DEFINED ENV{CI} AND NOT "$ENV{CI}" STREQUAL "" AND NOT "$ENV{CI}" STREQUAL "false")
        set(YERBAS_CUDA_ARCHITECTURES_DEFAULT "52;60;61;70;75;80;86;89;90")
    else()
        set(YERBAS_CUDA_ARCHITECTURES_DEFAULT "native")
    endif()
    set(YERBAS_CUDA_ARCHITECTURES "${YERBAS_CUDA_ARCHITECTURES_DEFAULT}" CACHE STRING
        "CUDA compute capabilities built into Yerbas-Miner (native locally; broad set in CI)")
    message(STATUS "Yerbas CUDA backend enabled with compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "Yerbas CUDA architectures: ${YERBAS_CUDA_ARCHITECTURES}")

    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(YERBAS_CUDA_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated_cuda")
    file(MAKE_DIRECTORY "${YERBAS_CUDA_GENERATED_DIR}")

    function(yerbas_generate_cuda_sph STEM SOURCE_FILE HEADER_FILE)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_cuda_sph.py"
                --stem "${STEM}"
                --source "${YERBAS_CORE_SRC}/crypto/${SOURCE_FILE}"
                --header "${YERBAS_CORE_SRC}/crypto/${HEADER_FILE}"
                --out-source "${YERBAS_CUDA_GENERATED_DIR}/${STEM}_device_impl.cuh"
                --out-header "${YERBAS_CUDA_GENERATED_DIR}/${STEM}_device_types.cuh"
                ${ARGN}
            RESULT_VARIABLE YERBAS_CUDA_SPH_RESULT
        )
        if(NOT YERBAS_CUDA_SPH_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to generate CUDA ${STEM} device source")
        endif()
    endfunction()

    yerbas_generate_cuda_sph(bmw bmw.c sph_bmw.h)
    yerbas_generate_cuda_sph(groestl groestl.c sph_groestl.h)
    yerbas_generate_cuda_sph(jh jh.c sph_jh.h)
    yerbas_generate_cuda_sph(skein skein.c sph_skein.h)
    yerbas_generate_cuda_sph(luffa luffa.c sph_luffa.h)
    yerbas_generate_cuda_sph(shavite shavite.c sph_shavite.h
        --inline-include "aes_helper.c=${YERBAS_CORE_SRC}/crypto/aes_helper.c")
    yerbas_generate_cuda_sph(simd simd.c sph_simd.h)
    yerbas_generate_cuda_sph(echo echo.c sph_echo.h
        --inline-include "aes_helper.c=${YERBAS_CORE_SRC}/crypto/aes_helper.c")
    yerbas_generate_cuda_sph(hamsi sph_hamsi.c sph_hamsi.h
        --inline-include "sph_hamsi_helper.c=${YERBAS_CORE_SRC}/crypto/sph_hamsi_helper.c")
    yerbas_generate_cuda_sph(fugue sph_fugue.c sph_fugue.h)
    yerbas_generate_cuda_sph(shabal sph_shabal.c sph_shabal.h)
    yerbas_generate_cuda_sph(whirlpool sph_whirlpool.c sph_whirlpool.h
        --inline-include-all "md_helper.c=${YERBAS_CORE_SRC}/crypto/md_helper.c")

    function(yerbas_generate_cuda_cn_hash NAMESPACE SOURCE_FILE HEADER_FILE OUTPUT_FILE)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/generate_cuda_cn_hash.py"
                --namespace "${NAMESPACE}"
                --source "${YERBAS_CORE_SRC}/cryptonote/${SOURCE_FILE}"
                --header "${YERBAS_CORE_SRC}/cryptonote/${HEADER_FILE}"
                --out "${YERBAS_CUDA_GENERATED_DIR}/${OUTPUT_FILE}"
            RESULT_VARIABLE YERBAS_CUDA_CN_RESULT
        )
        if(NOT YERBAS_CUDA_CN_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to generate CUDA CryptoNight primitive ${NAMESPACE}")
        endif()
    endfunction()

    yerbas_generate_cuda_cn_hash(cn_blake256 c_blake256.c c_blake256.h cn_blake256_device.cuh)
    yerbas_generate_cuda_cn_hash(cn_groestl c_groestl.c c_groestl.h cn_groestl_device.cuh)
    yerbas_generate_cuda_cn_hash(cn_jh c_jh.c c_jh.h cn_jh_device.cuh)
    yerbas_generate_cuda_cn_hash(cn_skein c_skein.c c_skein.h cn_skein_device.cuh)

    target_sources(yerbas-miner PRIVATE
        src/cuda/cuda_backend.cu
        src/cuda/blake_validation.cu
        src/cuda/skein_validation.cu
        src/cuda/sph_batch_validation.cu
    )
    target_include_directories(yerbas-miner PRIVATE
        "${YERBAS_CUDA_GENERATED_DIR}"
        "${YERBAS_CORE_SRC}/crypto"
        "${YERBAS_CORE_SRC}/cryptonote"
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

    add_executable(ghostrider-equivalence tests/ghostrider_equivalence.cpp)
    target_include_directories(ghostrider-equivalence PRIVATE src ${YERBAS_CORE_SRC})
    target_link_libraries(ghostrider-equivalence PRIVATE yerbas-ghostrider-reference)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_compile_options(ghostrider-equivalence PRIVATE
            -include "${CMAKE_CURRENT_SOURCE_DIR}/cmake/glibc_endian_compat.h"
        )
    endif()
    add_test(NAME ghostrider-equivalence COMMAND ghostrider-equivalence)

    if(YERBAS_CUDA_AVAILABLE)
        add_executable(cuda-keccak-validation
            tests/cuda_keccak_validation.cpp
            src/cuda/cuda_backend.cu
            src/cuda/blake_validation.cu
            src/cuda/skein_validation.cu
            src/cuda/sph_batch_validation.cu
            src/cuda/cryptonight/cn_validation.cu
        )
        target_include_directories(cuda-keccak-validation PRIVATE
            src
            tests
            "${YERBAS_CUDA_GENERATED_DIR}"
            "${YERBAS_CORE_SRC}/crypto"
            "${YERBAS_CORE_SRC}/cryptonight"
        )
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
