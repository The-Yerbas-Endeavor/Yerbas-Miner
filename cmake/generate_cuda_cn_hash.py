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


def rewrite_core_groestl_for_cuda(source: str) -> str:
    """Remove c_groestl.c's byte-buffer -> uint32_t aliasing from Transform().

    Yerbas Core's CPU implementation casts arbitrary byte input directly to
    uint32_t*. That works on the x86 consensus path but is undefined C aliasing
    and can be optimized differently by NVCC. Reconstruct the 16 little-endian
    message words explicitly before F512 so CUDA observes exactly the x86 byte
    interpretation for both direct message blocks and ctx->buffer padding.
    """
    needle = '    F512(ctx->chaining,(uint32_t*)input);'
    replacement = '''    uint32_t m_words[2*COLS512];
    for (int wi = 0; wi < 2*COLS512; ++wi) {
      const uint8_t *p = input + 4 * wi;
      m_words[wi] = ((uint32_t)p[0]) |
                    ((uint32_t)p[1] << 8) |
                    ((uint32_t)p[2] << 16) |
                    ((uint32_t)p[3] << 24);
    }
    F512(ctx->chaining, m_words);'''
    if needle not in source:
        raise RuntimeError("could not locate Core Groestl Transform input cast")
    return source.replace(needle, replacement, 1)


def helpers() -> str:
    return r'''
typedef unsigned char BitSequence;
typedef unsigned long long DataLength;

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
    if a.namespace == 'cn_groestl':
        source = rewrite_core_groestl_for_cuda(source)
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
