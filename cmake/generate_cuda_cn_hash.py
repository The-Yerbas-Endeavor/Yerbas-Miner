#!/usr/bin/env python3
"""Adapt pinned Yerbas Core CryptoNight extra-hash C code for CUDA device use.

This generator is intentionally mechanical: the pinned Core source remains the
algorithm source of truth. It converts file-scope C functions/tables to private
CUDA device symbols and replaces the small libc memory surface with device-safe
helpers.
"""
from __future__ import annotations
import argparse
import pathlib
import re


def strip_includes(text: str) -> str:
    return re.sub(r'(?m)^\s*#\s*include\s+[<\"].*[>\"]\s*$', '', text)


def strip_c_linkage(text: str) -> str:
    text = re.sub(r'(?ms)^\s*#\s*if(?:def|\s+defined\s*\()__cplusplus\)?\s*\n\s*extern\s+"C"\s*\{\s*\n\s*#\s*endif\s*', '', text)
    text = re.sub(r'(?ms)^\s*#\s*if(?:def|\s+defined\s*\()__cplusplus\)?\s*\n\s*\}\s*\n\s*#\s*endif\s*', '', text)
    return text


def inline_core_dependencies(source: str, source_path: pathlib.Path) -> str:
    """Inline small local headers whose data definitions are required on device."""
    for name in ("groestl_tables.h",):
        include_re = re.compile(rf'(?m)^\s*#\s*include\s+[\"]{re.escape(name)}[\"]\s*$')
        if include_re.search(source):
            dependency = source_path.parent / name
            if not dependency.exists():
                raise FileNotFoundError(f"required Core dependency not found: {dependency}")
            source = include_re.sub(dependency.read_text(encoding="utf-8"), source)
    return source


def rewrite_groestl_word_aliasing(source: str, source_path: pathlib.Path) -> str:
    """Remove x86-only uint32_t aliasing from the CryptoNote Groestl rounds.

    c_groestl.c mutates a byte buffer through ``uint32_t *x32``. That is fine
    for the original x86 C build but is not a safe assumption for CUDA local
    memory. Convert those fixed round-constant updates to explicit little-endian
    loads/stores while leaving the Core round structure and T tables unchanged.
    """
    if source_path.name != "c_groestl.c":
        return source

    source = re.sub(r'(?m)^\s*uint32_t\s*\*\s*x32\s*=\s*\(uint32_t\s*\*\)x\s*;\s*$', '', source)

    # x32[n] ^= expression;
    source = re.sub(
        r'(?m)^(\s*)x32\[\s*(\d+)\s*\]\s*\^=\s*([^;]+);',
        lambda m: (
            f"{m.group(1)}yerbas_cn_store32le(x + {int(m.group(2)) * 4}, "
            f"yerbas_cn_load32le(x + {int(m.group(2)) * 4}) ^ ({m.group(3).strip()}));"
        ),
        source,
    )

    # x32[n] = ~x32[n];
    source = re.sub(
        r'(?m)^(\s*)x32\[\s*(\d+)\s*\]\s*=\s*~\s*x32\[\s*\2\s*\]\s*;',
        lambda m: (
            f"{m.group(1)}yerbas_cn_store32le(x + {int(m.group(2)) * 4}, "
            f"~yerbas_cn_load32le(x + {int(m.group(2)) * 4}));"
        ),
        source,
    )
    return source


def helpers() -> str:
    return r'''
typedef unsigned char BitSequence;
typedef unsigned long long DataLength;

__device__ __forceinline__ uint32_t yerbas_cn_load32le(const unsigned char *p)
{
    return ((uint32_t)p[0]) |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}
__device__ __forceinline__ void yerbas_cn_store32le(unsigned char *p, uint32_t v)
{
    p[0] = (unsigned char)v;
    p[1] = (unsigned char)(v >> 8);
    p[2] = (unsigned char)(v >> 16);
    p[3] = (unsigned char)(v >> 24);
}
__device__ __forceinline__ void *yerbas_cn_memcpy(void *dst, const void *src, size_t n)
{
    unsigned char *d = static_cast<unsigned char *>(dst);
    const unsigned char *s = static_cast<const unsigned char *>(src);
    for (size_t i = 0; i < n; ++i) d[i] = s[i];
    return dst;
}
__device__ __forceinline__ void *yerbas_cn_memset(void *dst, int value, size_t n)
{
    unsigned char *d = static_cast<unsigned char *>(dst);
    for (size_t i = 0; i < n; ++i) d[i] = static_cast<unsigned char>(value);
    return dst;
}
#define memcpy yerbas_cn_memcpy
#define memset yerbas_cn_memset

'''


def decorate_prototypes(header: str) -> str:
    pattern = re.compile(
        r'(?m)^(?!\s*(?:typedef|struct|enum|union|#))'
        r'(\s*(?:void|int|uint8_t|uint32_t|uint64_t|size_t|HashReturn)\s+\**\s*)'
        r'([A-Za-z_][A-Za-z0-9_]*)\s*(\([^;{}]*\)\s*;)'
    )
    return pattern.sub(r'__device__ static \1\2\3', header)


def decorate_source(source: str) -> str:
    source = re.sub(r'(?m)^static\s+', '__device__ static ', source)
    pattern = re.compile(
        r'(?m)^(?!__device__)'
        r'((?:void|int|uint8_t|uint32_t|uint64_t|size_t|HashReturn)\s+\**\s*)'
        r'([A-Za-z_][A-Za-z0-9_]*)\s*(\([^;]*\)\s*\{)'
    )
    source = pattern.sub(r'__device__ static \1\2\3', source)
    source = re.sub(
        r'(?m)^(?!__device__)(const\s+(?:uint8_t|uint32_t|uint64_t|int|unsigned char|unsigned int)\s+[A-Za-z_][A-Za-z0-9_]*\s*\[)',
        r'__device__ static \1', source)
    return source


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--namespace', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--header', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()

    source_path = pathlib.Path(a.source)
    header = pathlib.Path(a.header).read_text(encoding='utf-8')
    source = source_path.read_text(encoding='utf-8')
    source = inline_core_dependencies(source, source_path)
    source = rewrite_groestl_word_aliasing(source, source_path)
    header = decorate_prototypes(strip_includes(strip_c_linkage(header)))
    source = decorate_source(strip_includes(strip_c_linkage(source)))

    out = (
        '/* Generated from pinned Yerbas Core CryptoNight primitive. Do not edit. */\n'
        '#pragma once\n#include <stddef.h>\n#include <stdint.h>\n'
        f'namespace yerbas::cuda::cryptonight::{a.namespace} {{\n'
        + helpers() + header + '\n' + source +
        '\n#undef memcpy\n#undef memset\n'
        f'}} // namespace yerbas::cuda::cryptonight::{a.namespace}\n'
    )
    path = pathlib.Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(out, encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
