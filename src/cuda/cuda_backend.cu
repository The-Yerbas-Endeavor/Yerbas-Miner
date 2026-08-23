// Split into implementation fragments so the CUDA backend can be maintained
// safely through GitHub's bounded file-edit interface. The fragments are
// textual includes and compile as one CUDA translation unit.
#include "cuda/generated/cuda_backend_part1.inc"
#include "cuda/generated/cuda_backend_part2a.inc"
#include "cuda/generated/cuda_backend_part2b.inc"

namespace yerbas::cuda {

#include "cuda/generated/cuda_backend_part3.inc"
#include "cuda/generated/cuda_backend_part4.inc"

} // namespace yerbas::cuda
