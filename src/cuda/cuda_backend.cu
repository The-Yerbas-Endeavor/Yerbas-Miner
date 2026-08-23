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

// Optional parity-gated cooperative loop probe. This wrapper calls the production
// tuner first and only runs when YERBAS_CUDA_COOP_PROBE=1; production dispatch is
// left unchanged until the cooperative measurements prove worthwhile.
#include "cuda/generated/cuda_backend_coop_probe.inc"

#include "cuda/generated/cuda_backend_part3.inc"
#include "cuda/generated/cuda_backend_part4.inc"
