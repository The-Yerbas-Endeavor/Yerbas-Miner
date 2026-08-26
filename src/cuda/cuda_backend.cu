#include "miner.h"

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

#define autotune_cn_geometries autotune_cn_geometries_legacy
#define launch_split_cryptonight_variant launch_split_cryptonight_variant_legacy
#define launch_split_cryptonight launch_split_cryptonight_legacy
#include "cuda/generated/cuda_backend_part2b.inc"
#undef launch_split_cryptonight
#undef launch_split_cryptonight_variant
#undef autotune_cn_geometries

#include "cuda/generated/cuda_backend_batch_tune.inc"

#define autotune_cn_geometries autotune_cn_geometries_prod
#include "cuda/generated/cuda_backend_prod_tune.inc"
#undef autotune_cn_geometries

#define autotune_cn_geometries autotune_cn_geometries_coop_fixed128
#include "cuda/generated/cuda_backend_coop_probe.inc"
#undef autotune_cn_geometries

#include "cuda/generated/cuda_backend_coop_v3.inc"
#include "cuda/generated/cuda_backend_coop_v4.inc"
#include "cuda/generated/cuda_backend_coop_v5.inc"
#include "cuda/generated/cuda_backend_coop_v6.inc"
#include "cuda/generated/cuda_backend_coop_v7.inc"

#define autotune_cn_geometries autotune_cn_geometries_coop
#include "cuda/generated/cuda_backend_coop_tune.inc"
#undef autotune_cn_geometries

#include "cuda/generated/cuda_backend_coop_impl_tune.inc"
#include "cuda/generated/cuda_backend_cn_fast_inner_profile_v2.inc"
#include "cuda/generated/cuda_backend_cn_fast_v4_tune.inc"
#include "cuda/generated/cuda_backend_cn_fast_v5_tune.inc"
#include "cuda/generated/cuda_backend_cn_fast_v6_tune.inc"
#include "cuda/generated/cuda_backend_cn_fast_v7_tune.inc"

#define autotune_cn_geometries autotune_cn_geometries_cnfast_internal
#include "cuda/generated/cuda_backend_cn_fast_tune.inc"
#undef autotune_cn_geometries

#include "cuda/generated/cuda_backend_cn_fast_microcache.inc"
#include "cuda/generated/cuda_backend_cn_fast_microtune.inc"
#include "cuda/generated/cuda_backend_cn_fast_microconfirm.inc"
#include "cuda/generated/cuda_backend_cn_fast_dispatch.inc"
#include "cuda/generated/cuda_backend_coop_dispatch.inc"

#include "cuda/generated/cuda_backend_part3.inc"
#include "cuda/generated/cuda_backend_part4.inc"
