#!/usr/bin/env python3
"""Temporarily inject CUDA first-iteration trace printing into the validator.

This edits tests/cuda_keccak_validation.cpp in the local checkout only. The
runner restores that file after execution, so the repository source remains
clean after the trace run.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", default="tests/cuda_keccak_validation.cpp")
    args = parser.parse_args()
    path = Path(args.source)
    if not path.is_file():
        raise SystemExit(f"missing validator source: {path}")

    text = path.read_text()
    marker = "YERBAS_CUDA_FIRST_ITER_TRACE_PRINT_V1"
    if marker in text:
        print(f"CUDA first-iteration trace printing already installed in {path}")
        return

    anchor = (
        "    const auto gpu_cp = yerbas::cuda::cryptonight::validation_checkpoints(\n"
        "        0, 0, stage_input.data(), stage_input.size());\n"
    )
    if text.count(anchor) != 1:
        raise SystemExit("could not locate gpu_cp checkpoint call in validator")

    block = anchor + r'''

    /* YERBAS_CUDA_FIRST_ITER_TRACE_PRINT_V1 */
    std::cout << "CUDA initial a:                " << hex_string(gpu_cp.first_initial_a) << "\n"
              << "CUDA initial b[0..15]:         " << hex_string(gpu_cp.first_initial_b) << "\n"
              << "CUDA j1: " << gpu_cp.first_j1 << "\n"
              << "CUDA slot1 before AES:         " << hex_string(gpu_cp.first_slot1_before_aes) << "\n"
              << "CUDA c after AES:              " << hex_string(gpu_cp.first_c_after_aes) << "\n"
              << "CUDA slot1 after c xor b:      " << hex_string(gpu_cp.first_slot1_after_xor) << "\n"
              << "CUDA slot1 after VARIANT1_1:   " << hex_string(gpu_cp.first_slot1_after_variant1) << "\n"
              << "CUDA j2: " << gpu_cp.first_j2 << "\n"
              << "CUDA slot2 before MUL:         " << hex_string(gpu_cp.first_slot2_before_mul) << "\n"
              << "CUDA t:                        " << hex_string(gpu_cp.first_t) << "\n"
              << std::hex << std::setfill('0')
              << "CUDA mul hi|lo: " << std::setw(16) << gpu_cp.first_mul_hi
              << std::setw(16) << gpu_cp.first_mul_lo << std::dec << "\n"
              << "CUDA slot2 after add:          " << hex_string(gpu_cp.first_slot2_after_add) << "\n"
              << "CUDA a after xor t:            " << hex_string(gpu_cp.first_a_after_xor) << "\n"
              << "CUDA slot2 after VARIANT1_2:   " << hex_string(gpu_cp.first_slot2_after_variant1_2) << "\n";
'''

    path.write_text(text.replace(anchor, block, 1))
    print(f"Installed CUDA first-iteration trace printing in {path}")


if __name__ == "__main__":
    main()
