// Split into implementation fragments so the CUDA backend can be maintained
// safely through GitHub's bounded file-edit interface. The fragments are
// textual includes and compile as one CUDA translation unit.
//
// part1 opens namespace yerbas::cuda and its anonymous implementation namespace.
// part2b closes the anonymous namespace. part4 closes namespace yerbas::cuda.
// Keep all fragments in the same lexical scope; do not wrap part3/part4 in a
// second namespace yerbas::cuda block.
#include "cuda/generated/cuda_backend_part1.inc"
#include "cuda/generated/cuda_backend_part2a.inc"

// Preserve the established small-sample tuner and split CryptoNight dispatch as
// fallbacks. The public dispatch is replaced below only after cooperative mode
// has passed parity and performance selection.
#define autotune_cn_geometries autotune_cn_geometries_legacy
#define launch_split_cryptonight_variant launch_split_cryptonight_variant_legacy
#define launch_split_cryptonight launch_split_cryptonight_legacy
#include "cuda/generated/cuda_backend_part2b.inc"
#undef launch_split_cryptonight
#undef launch_split_cryptonight_variant
#undef autotune_cn_geometries

// Preserve production-batch thread/unroll tuning as the second-stage implementation.
#define autotune_cn_geometries autotune_cn_geometries_prod
#include "cuda/generated/cuda_backend_prod_tune.inc"
#undef autotune_cn_geometries

// Keep the original cooperative probe implementation available for its kernels
// and parity helpers. Its wrapper remains diagnostic-only.
#define autotune_cn_geometries autotune_cn_geometries_coop_fixed128
#include "cuda/generated/cuda_backend_coop_probe.inc"
#undef autotune_cn_geometries

// Final-stage production selector: validates complete CryptoNight output, sweeps
// architecture-neutral cooperative geometries, applies a 2% speedup gate, and
// persists the decision per GPU/driver/runtime/batch.
#include "cuda/generated/cuda_backend_coop_tune.inc"

// Public CryptoNight dispatch: selected variants use cooperative mode; all other
// cases transparently call the established production implementation.
#include "cuda/generated/cuda_backend_coop_dispatch.inc"

#include "cuda/generated/cuda_backend_part3.inc"
#include "cuda/generated/cuda_backend_part4.inc"
