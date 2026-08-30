#pragma once

#include <cstdint>
#include <string>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

namespace yerbas::cpu {

struct X86Features {
    bool available{false};
    bool aes{false};
    bool avx{false};
    bool avx2{false};
    bool bmi2{false};
    bool sse42{false};
    bool osxsave{false};
    bool ymm_state{false};
};

inline X86Features detect_x86_features() noexcept
{
    X86Features f{};

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int regs[4]{};
    __cpuid(regs, 0);
    const int max_leaf = regs[0];
    if (max_leaf < 1) return f;

    __cpuidex(regs, 1, 0);
    const std::uint32_t ecx = static_cast<std::uint32_t>(regs[2]);
    f.available = true;
    f.sse42 = (ecx & (1U << 20)) != 0U;
    f.aes = (ecx & (1U << 25)) != 0U;
    f.osxsave = (ecx & (1U << 27)) != 0U;
    const bool avx_hw = (ecx & (1U << 28)) != 0U;

    if (f.osxsave) {
        const unsigned __int64 xcr0 = _xgetbv(0);
        f.ymm_state = (xcr0 & 0x6U) == 0x6U;
    }
    f.avx = avx_hw && f.ymm_state;

    if (max_leaf >= 7) {
        __cpuidex(regs, 7, 0);
        const std::uint32_t ebx = static_cast<std::uint32_t>(regs[1]);
        f.avx2 = f.avx && (ebx & (1U << 5)) != 0U;
        f.bmi2 = (ebx & (1U << 8)) != 0U;
    }
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    f.available = true;
    f.aes = __builtin_cpu_supports("aes");
    f.avx = __builtin_cpu_supports("avx");
    f.avx2 = __builtin_cpu_supports("avx2");
    f.bmi2 = __builtin_cpu_supports("bmi2");
    f.sse42 = __builtin_cpu_supports("sse4.2");
    // GCC/Clang builtins already account for OS AVX state support.
    f.osxsave = f.avx;
    f.ymm_state = f.avx;
#endif

    return f;
}

inline const char* yes_no(bool value) noexcept { return value ? "yes" : "no"; }

} // namespace yerbas::cpu
