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

// Preserve the established small-sample tuner as the first-stage implementation.
#define autotune_cn_geometries autotune_cn_geometries_legacy
#include "cuda/generated/cuda_backend_part2b.inc"
#undef autotune_cn_geometries

// Preserve production-batch thread/unroll tuning as the second-stage implementation.
#define autotune_cn_geometries autotune_cn_geometries_prod
#include "cuda/generated/cuda_backend_prod_tune.inc"
#undef autotune_cn_geometries

// Keep the original cooperative probe implementation available for its parity
// kernels/helpers, but rename its wrapper so the architecture-neutral geometry
// sweep below becomes the public autotune wrapper.
#define autotune_cn_geometries autotune_cn_geometries_coop_fixed128
#include "cuda/generated/cuda_backend_coop_probe.inc"
#undef autotune_cn_geometries

// Probe the parity-proven cooperative kernel at several block sizes and compare
// against the actual tuned production single-hash path. Production dispatch is
// still unchanged until the measurements establish a safe per-device selector.
#include "cuda/generated/cuda_backend_coop_tune.inc"

#include "cuda/generated/cuda_backend_part3.inc"
#include "cuda/generated/cuda_backend_part4.inc"
