/* AES-NI/AVX2 CryptoNight implementation used only after runtime CPUID dispatch. */
#define YERBAS_FORCE_AESNI 1
#define YERBAS_FORCE_AVX2 1
#define YERBAS_FORCE_BMI2 1
#define YERBAS_FORCE_SSE42 1
#define yerbas_cn_reuse_backend yerbas_cn_reuse_backend_aes_avx2
#define yerbas_cn_reuse_compile_features yerbas_cn_reuse_compile_features_aes_avx2
#define yerbas_cn_slow_hash_reuse yerbas_cn_slow_hash_reuse_aes_avx2
#include "slow_hash_reuse.c"
