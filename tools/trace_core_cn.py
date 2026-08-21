#!/usr/bin/env python3
"""Inject first-iteration trace prints into the fetched pinned Core slow-hash.c.

Run after `cmake -S . -B build-cuda` and before rebuilding the validator.
This edits only build-cuda/_deps/yerbas_core-src, never the Yerbas-Miner source tree.
Re-running CMake configure restores the pinned Core file first.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"trace patch failed for {label}: expected 1 match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "slow_hash",
        nargs="?",
        default="build-cuda/_deps/yerbas_core-src/src/cryptonote/slow-hash.c",
    )
    args = parser.parse_args()
    path = Path(args.slow_hash)
    if not path.is_file():
        raise SystemExit(f"missing fetched Core source: {path}")

    text = path.read_text()
    marker = "YERBAS_CN_FIRST_ITER_TRACE_V3"
    if marker in text:
        print(f"Core first-iteration trace already installed in {path}")
        return

    anchor = "static void yerbas_cn_debug_print_hex(const char* label, const uint8_t* p, size_t n) {"
    if anchor not in text:
        raise SystemExit(
            "current CMake validator instrumentation is missing; run `cmake -S . -B build-cuda` first"
        )

    text = replace_once(text, anchor, "/* YERBAS_CN_FIRST_ITER_TRACE_V3 */\n" + anchor, "marker")

    text = replace_once(
        text,
        "  for (i = 0; i < iterations; i++) {\n    /* Dependency chain: address -> read value ------+",
        "  for (i = 0; i < iterations; i++) {\n"
        "    if (g_yerbas_cn_debug_capture && i == 0) {\n"
        "      yerbas_cn_debug_print_hex(\"Raw Core initial a:             \", a, 16);\n"
        "      yerbas_cn_debug_print_hex(\"Raw Core initial b[0..15]:      \", b, 16);\n"
        "    }\n"
        "    /* Dependency chain: address -> read value ------+",
        "initial a/b",
    )

    text = replace_once(
        text,
        "    j = e2i(a, aes_rounds);\n    aesb_single_round(&long_state[j * AES_BLOCK_SIZE], c, a);",
        "    j = e2i(a, aes_rounds);\n"
        "    if (g_yerbas_cn_debug_capture && i == 0) {\n"
        "      printf(\"Raw Core j1: %zu\\n\", j);\n"
        "      yerbas_cn_debug_print_hex(\"Raw Core slot1 before AES:      \", &long_state[j * AES_BLOCK_SIZE], 16);\n"
        "    }\n"
        "    aesb_single_round(&long_state[j * AES_BLOCK_SIZE], c, a);\n"
        "    if (g_yerbas_cn_debug_capture && i == 0) {\n"
        "      uint8_t expected_xor[16];\n"
        "      size_t xk;\n"
        "      yerbas_cn_debug_print_hex(\"Raw Core c after AES:            \", c, 16);\n"
        "      yerbas_cn_debug_print_hex(\"Raw Core b immediately pre-XOR: \", b, 16);\n"
        "      printf(\"Raw Core c ptr: %p b ptr: %p dst ptr: %p\\n\", (void*)c, (void*)b, (void*)&long_state[j * AES_BLOCK_SIZE]);\n"
        "      printf(\"Raw Core c64: %016llx %016llx\\n\", (unsigned long long)((uint64_t*)c)[0], (unsigned long long)((uint64_t*)c)[1]);\n"
        "      printf(\"Raw Core b64: %016llx %016llx\\n\", (unsigned long long)((uint64_t*)b)[0], (unsigned long long)((uint64_t*)b)[1]);\n"
        "      for (xk = 0; xk < 16; ++xk) expected_xor[xk] = c[xk] ^ b[xk];\n"
        "      yerbas_cn_debug_print_hex(\"Raw Core bytewise c xor b:      \", expected_xor, 16);\n"
        "    }",
        "first AES",
    )

    text = replace_once(
        text,
        "    xor_blocks_dst(c, b, &long_state[j * AES_BLOCK_SIZE]);\n    VARIANT1_1((uint8_t*)&long_state[j * AES_BLOCK_SIZE]);",
        "    if (g_yerbas_cn_debug_capture && i == 0) {\n"
        "      uint64_t dbg_c0 = ((uint64_t*)c)[0];\n"
        "      uint64_t dbg_c1 = ((uint64_t*)c)[1];\n"
        "      uint64_t dbg_b0 = ((uint64_t*)b)[0];\n"
        "      uint64_t dbg_b1 = ((uint64_t*)b)[1];\n"
        "      printf(\"Raw Core xor64 expected: %016llx %016llx\\n\", (unsigned long long)(dbg_c0 ^ dbg_b0), (unsigned long long)(dbg_c1 ^ dbg_b1));\n"
        "    }\n"
        "    xor_blocks_dst(c, b, &long_state[j * AES_BLOCK_SIZE]);\n"
        "    if (g_yerbas_cn_debug_capture && i == 0) {\n"
        "      printf(\"Raw Core dst64 actual:  %016llx %016llx\\n\", (unsigned long long)((uint64_t*)&long_state[j * AES_BLOCK_SIZE])[0], (unsigned long long)((uint64_t*)&long_state[j * AES_BLOCK_SIZE])[1]);\n"
        "      yerbas_cn_debug_print_hex(\"Raw Core xor_blocks_dst result: \", &long_state[j * AES_BLOCK_SIZE], 16);\n"
        "    }\n"
        "    VARIANT1_1((uint8_t*)&long_state[j * AES_BLOCK_SIZE]);\n"
        "    if (g_yerbas_cn_debug_capture && i == 0)\n"
        "      yerbas_cn_debug_print_hex(\"Raw Core slot1 after VARIANT1_1: \", &long_state[j * AES_BLOCK_SIZE], 16);",
        "first scratch write",
    )

    text = replace_once(
        text,
        "    j = e2i(c, aes_rounds);\n\n    uint64_t* dst = (uint64_t*)&long_state[j * AES_BLOCK_SIZE];",
        "    j = e2i(c, aes_rounds);\n"
        "    if (g_yerbas_cn_debug_capture && i == 0) {\n"
        "      printf(\"Raw Core j2: %zu\\n\", j);\n"
        "      yerbas_cn_debug_print_hex(\"Raw Core slot2 before MUL:      \", &long_state[j * AES_BLOCK_SIZE], 16);\n"
        "    }\n\n"
        "    uint64_t* dst = (uint64_t*)&long_state[j * AES_BLOCK_SIZE];",
        "second address",
    )

    text = replace_once(
        text,
        "    t[0] = dst[0];\n    t[1] = dst[1];",
        "    t[0] = dst[0];\n"
        "    t[1] = dst[1];\n"
        "    if (g_yerbas_cn_debug_capture && i == 0)\n"
        "      yerbas_cn_debug_print_hex(\"Raw Core t:                     \", (const uint8_t*)t, 16);",
        "t read",
    )

    text = replace_once(
        text,
        "    uint64_t hi;\n    uint64_t lo = mul128(((uint64_t*)c)[0], t[0], &hi);",
        "    uint64_t hi;\n"
        "    uint64_t lo = mul128(((uint64_t*)c)[0], t[0], &hi);\n"
        "    if (g_yerbas_cn_debug_capture && i == 0)\n"
        "      printf(\"Raw Core mul hi|lo: %016llx%016llx\\n\", (unsigned long long)hi, (unsigned long long)lo);",
        "multiply",
    )

    text = replace_once(
        text,
        "    dst[0] = ((uint64_t*)a)[0];\n    dst[1] = ((uint64_t*)a)[1];",
        "    dst[0] = ((uint64_t*)a)[0];\n"
        "    dst[1] = ((uint64_t*)a)[1];\n"
        "    if (g_yerbas_cn_debug_capture && i == 0)\n"
        "      yerbas_cn_debug_print_hex(\"Raw Core slot2 after add:        \", (const uint8_t*)dst, 16);",
        "slot after add",
    )

    text = replace_once(
        text,
        "    ((uint64_t*)a)[0] ^= t[0];\n    ((uint64_t*)a)[1] ^= t[1];\n\n    VARIANT1_2((uint8_t*)&long_state[j * AES_BLOCK_SIZE]);",
        "    ((uint64_t*)a)[0] ^= t[0];\n"
        "    ((uint64_t*)a)[1] ^= t[1];\n"
        "    if (g_yerbas_cn_debug_capture && i == 0)\n"
        "      yerbas_cn_debug_print_hex(\"Raw Core a after xor t:          \", a, 16);\n\n"
        "    VARIANT1_2((uint8_t*)&long_state[j * AES_BLOCK_SIZE]);\n"
        "    if (g_yerbas_cn_debug_capture && i == 0)\n"
        "      yerbas_cn_debug_print_hex(\"Raw Core slot2 after VARIANT1_2: \", &long_state[j * AES_BLOCK_SIZE], 16);",
        "variant1 tweak",
    )

    path.write_text(text)
    print(f"Installed Core first-iteration trace in {path}")


if __name__ == "__main__":
    main()
