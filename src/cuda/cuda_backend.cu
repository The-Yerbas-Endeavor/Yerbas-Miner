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
// The production tuner included immediately afterwards calls it, then re-tunes
// only the CN memory-loop threads/unroll pair at the real configured batch size.
#define autotune_cn_geometries autotune_cn_geometries_legacy
#include "cuda/generated/cuda_backend_part2b.inc"
#undef autotune_cn_geometries
#include "cuda/generated/cuda_backend_prod_tune.inc"

#include "cuda/generated/cuda_backend_part3.inc"
#include "cuda/generated/cuda_backend_part4.inc"
