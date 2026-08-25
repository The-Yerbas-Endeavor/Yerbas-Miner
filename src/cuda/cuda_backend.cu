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

// Select the production mining batch before per-batch geometry/backend tuning.
// The explicit YERBAS_CUDA_ACTIVE_BATCH environment variable remains an override.
#include "cuda/generated/cuda_backend_batch_tune.inc"

// Preserve production-batch thread/unroll tuning as the second-stage implementation.
#define autotune_cn_geometries autotune_cn_geometries_prod
#include "cuda/generated/cuda_backend_prod_tune.inc"
#undef autotune_cn_geometries

// Keep the original cooperative probe implementation available for its kernels
// and parity helpers. Its wrapper remains diagnostic-only.
#define autotune_cn_geometries autotune_cn_geometries_coop_fixed128
#include "cuda/generated/cuda_backend_coop_probe.inc"
#undef autotune_cn_geometries

// Cooperative production selector remains the broad all-variant tuner, but its
// wrapper is preserved so CN-Fast can get one final backend-family comparison.
#define autotune_cn_geometries autotune_cn_geometries_coop
#include "cuda/generated/cuda_backend_coop_tune.inc"
#undef autotune_cn_geometries

// Final CN-Fast selector compares the existing single, dual-state, and coop4
// families at the real production batch size, validates parity, and caches the
// winner only when it clears a 3% improvement gate. Preserve its default wrapper
// under an internal name; the guarded public wrapper below protects an existing
// production unroll-4 winner from the intentionally narrower challenger set.
#define autotune_cn_geometries autotune_cn_geometries_cnfast_internal
#include "cuda/generated/cuda_backend_cn_fast_tune.inc"
#undef autotune_cn_geometries

// Persist setup/coop/final geometry independently per GPU, driver/runtime and
// active batch. Forced benchmark/retune modes bypass this cache.
#include "cuda/generated/cuda_backend_cn_fast_microcache.inc"

// Focused CN-Fast phase/microkernel A/B tuner. It runs only for the explicit
// --benchmark-cn-fast path and can refine setup, coop4 loop, and final geometry
// after the backend family winner has been established.
#include "cuda/generated/cuda_backend_cn_fast_microtune.inc"

// Direct repeated setup->loop->final timing confirms that a phase-sweep winner
// improves the real whole CN-Fast stage before it is persisted for production.
#include "cuda/generated/cuda_backend_cn_fast_microconfirm.inc"

#include "cuda/generated/cuda_backend_cn_fast_dispatch.inc"

// Public CryptoNight dispatch: selected variants use cooperative mode; all other
// cases transparently call the established production implementation.
#include "cuda/generated/cuda_backend_coop_dispatch.inc"

#include "cuda/generated/cuda_backend_part3.inc"
#include "cuda/generated/cuda_backend_part4.inc"
