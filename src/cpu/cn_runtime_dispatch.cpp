#include <cstddef>
#include <cstdint>
#include <iostream>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

extern "C" {
void yerbas_cn_slow_hash_reuse_portable(const char*, char*, std::uint32_t, int,
                                        std::uint32_t, std::uint32_t, std::size_t);
void yerbas_cn_slow_hash_reuse_aes_avx2(const char*, char*, std::uint32_t, int,
                                        std::uint32_t, std::uint32_t, std::size_t);
const char* yerbas_cn_reuse_backend_portable(void);
const char* yerbas_cn_reuse_backend_aes_avx2(void);
}

namespace {

enum class Backend { Portable, AesAvx2 };

bool cpu_has_aes_avx2_bmi2_sse42() noexcept
{
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("aes") &&
           __builtin_cpu_supports("avx2") &&
           __builtin_cpu_supports("bmi2") &&
           __builtin_cpu_supports("sse4.2");
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int regs[4]{};
    __cpuid(regs, 1);
    const bool aes = (regs[2] & (1 << 25)) != 0;
    const bool sse42 = (regs[2] & (1 << 20)) != 0;
    const bool osxsave = (regs[2] & (1 << 27)) != 0;
    const bool avx = (regs[2] & (1 << 28)) != 0;
    if (!aes || !sse42 || !osxsave || !avx) return false;
    const unsigned __int64 xcr0 = _xgetbv(0);
    if ((xcr0 & 0x6) != 0x6) return false;
    __cpuidex(regs, 7, 0);
    const bool avx2 = (regs[1] & (1 << 5)) != 0;
    const bool bmi2 = (regs[1] & (1 << 8)) != 0;
    return avx2 && bmi2;
#else
    return false;
#endif
}

Backend selected_backend() noexcept
{
    static const Backend backend = cpu_has_aes_avx2_bmi2_sse42()
        ? Backend::AesAvx2 : Backend::Portable;
    return backend;
}

void report_backend_once()
{
    static const bool reported = [] {
        const Backend backend = selected_backend();
        std::cout << "CPU CryptoNight runtime dispatch: selected="
                  << (backend == Backend::AesAvx2
                      ? yerbas_cn_reuse_backend_aes_avx2()
                      : yerbas_cn_reuse_backend_portable())
                  << " | fat-binary=yes\n";
        return true;
    }();
    (void)reported;
}

} // namespace

extern "C" void yerbas_cn_slow_hash_reuse(const char* input,
                                            char* output,
                                            std::uint32_t len,
                                            int variant,
                                            std::uint32_t page_size,
                                            std::uint32_t iterations,
                                            std::size_t aes_rounds)
{
    report_backend_once();
    if (selected_backend() == Backend::AesAvx2) {
        yerbas_cn_slow_hash_reuse_aes_avx2(input, output, len, variant,
                                           page_size, iterations, aes_rounds);
    } else {
        yerbas_cn_slow_hash_reuse_portable(input, output, len, variant,
                                           page_size, iterations, aes_rounds);
    }
}

extern "C" const char* yerbas_cn_runtime_backend(void)
{
    return selected_backend() == Backend::AesAvx2
        ? yerbas_cn_reuse_backend_aes_avx2()
        : yerbas_cn_reuse_backend_portable();
}
