#!/usr/bin/env python3
"""Generate CUDA-device copies of selected sphlib hash implementations.

The source of truth remains the Yerbas Core revision pinned by CMake. This
performs mechanical source adaptation so the exact pinned sphlib code can run
as CUDA device code without maintaining a second cryptographic implementation.
"""

from __future__ import annotations
import argparse
import pathlib
import re


def public_function_pattern(stem: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?ms)^\s*void\s+(sph_{re.escape(stem)}[A-Za-z0-9_]*)\s*\(([^;]*?)\)\s*;"
    )


def strip_c_linkage_wrappers(text: str) -> str:
    text = re.sub(
        r'(?ms)^\s*#\s*ifdef\s+__cplusplus\s*\n\s*extern\s+"C"\s*\{\s*\n\s*#\s*endif\s*',
        '', text,
    )
    text = re.sub(
        r'(?ms)^\s*#\s*ifdef\s+__cplusplus\s*\n\s*\}\s*\n\s*#\s*endif\s*',
        '', text,
    )
    return text


def strip_public_prototypes(header: str, stem: str) -> str:
    return public_function_pattern(stem).sub("", strip_c_linkage_wrappers(header))


def context_type_for(function_name: str, stem: str) -> str | None:
    # Whirlpool's md_helper-generated public entry points retain void *cc.
    # Typed pointers convert safely to void* in C++, while forcing a concrete
    # type here makes the macro-generated close wrappers incompatible.
    if stem == "whirlpool":
        return None
    bits = re.search(r"(224|256|384|512)", function_name)
    if bits:
        return f"sph_{stem}{bits.group(1)}_context"
    return None


def type_cc_parameter(signature: str, function_name: str, stem: str) -> str:
    context_type = context_type_for(function_name, stem)
    if context_type is None:
        return signature
    return re.sub(
        r"\bvoid\s*\*\s*cc\b",
        f"{context_type} *cc",
        signature,
        count=1,
    )


def device_public_prototypes(header: str, stem: str) -> str:
    declarations: list[str] = []
    for match in public_function_pattern(stem).finditer(header):
        name = match.group(1)
        params = type_cc_parameter(match.group(2), name, stem)
        declarations.append(f"__device__ static void {name}({params});")
    if not declarations:
        return ""
    return "/* generated device forward declarations */\n" + "\n".join(declarations) + "\n\n"


def type_public_definitions(source: str, stem: str) -> str:
    pattern = re.compile(
        rf"(?ms)(__device__\s+static\s+void\s+)"
        rf"(sph_{re.escape(stem)}[A-Za-z0-9_]*)"
        rf"(\s*\()(.*?)(\)\s*\{{)"
    )

    def repl(match: re.Match[str]) -> str:
        params = type_cc_parameter(match.group(4), match.group(2), stem)
        return match.group(1) + match.group(2) + match.group(3) + params + match.group(5)

    return pattern.sub(repl, source)


def device_endian_helpers() -> str:
    return r'''/* generated CUDA endian helpers */
__device__ __forceinline__ sph_u32 yerbas_cuda_dec32le(const void *src)
{
    const unsigned char *p = static_cast<const unsigned char *>(src);
    return (sph_u32)p[0] | ((sph_u32)p[1] << 8) | ((sph_u32)p[2] << 16) | ((sph_u32)p[3] << 24);
}
__device__ __forceinline__ sph_u32 yerbas_cuda_dec32be(const void *src)
{
    const unsigned char *p = static_cast<const unsigned char *>(src);
    return ((sph_u32)p[0] << 24) | ((sph_u32)p[1] << 16) | ((sph_u32)p[2] << 8) | (sph_u32)p[3];
}
__device__ __forceinline__ sph_u64 yerbas_cuda_dec64le(const void *src)
{
    const unsigned char *p = static_cast<const unsigned char *>(src);
    return (sph_u64)p[0] | ((sph_u64)p[1] << 8) | ((sph_u64)p[2] << 16) | ((sph_u64)p[3] << 24)
        | ((sph_u64)p[4] << 32) | ((sph_u64)p[5] << 40) | ((sph_u64)p[6] << 48) | ((sph_u64)p[7] << 56);
}
__device__ __forceinline__ sph_u64 yerbas_cuda_dec64be(const void *src)
{
    const unsigned char *p = static_cast<const unsigned char *>(src);
    return ((sph_u64)p[0] << 56) | ((sph_u64)p[1] << 48) | ((sph_u64)p[2] << 40) | ((sph_u64)p[3] << 32)
        | ((sph_u64)p[4] << 24) | ((sph_u64)p[5] << 16) | ((sph_u64)p[6] << 8) | (sph_u64)p[7];
}
__device__ __forceinline__ void yerbas_cuda_enc32le(void *dst, sph_u32 val)
{
    unsigned char *p = static_cast<unsigned char *>(dst);
    p[0]=(unsigned char)val; p[1]=(unsigned char)(val>>8); p[2]=(unsigned char)(val>>16); p[3]=(unsigned char)(val>>24);
}
__device__ __forceinline__ void yerbas_cuda_enc32be(void *dst, sph_u32 val)
{
    unsigned char *p = static_cast<unsigned char *>(dst);
    p[0]=(unsigned char)(val>>24); p[1]=(unsigned char)(val>>16); p[2]=(unsigned char)(val>>8); p[3]=(unsigned char)val;
}
__device__ __forceinline__ void yerbas_cuda_enc64le(void *dst, sph_u64 val)
{
    unsigned char *p = static_cast<unsigned char *>(dst);
    p[0]=(unsigned char)val; p[1]=(unsigned char)(val>>8); p[2]=(unsigned char)(val>>16); p[3]=(unsigned char)(val>>24);
    p[4]=(unsigned char)(val>>32); p[5]=(unsigned char)(val>>40); p[6]=(unsigned char)(val>>48); p[7]=(unsigned char)(val>>56);
}
__device__ __forceinline__ void yerbas_cuda_enc64be(void *dst, sph_u64 val)
{
    unsigned char *p = static_cast<unsigned char *>(dst);
    p[0]=(unsigned char)(val>>56); p[1]=(unsigned char)(val>>48); p[2]=(unsigned char)(val>>40); p[3]=(unsigned char)(val>>32);
    p[4]=(unsigned char)(val>>24); p[5]=(unsigned char)(val>>16); p[6]=(unsigned char)(val>>8); p[7]=(unsigned char)val;
}
#define sph_dec32le yerbas_cuda_dec32le
#define sph_dec32le_aligned yerbas_cuda_dec32le
#define sph_dec32be yerbas_cuda_dec32be
#define sph_dec32be_aligned yerbas_cuda_dec32be
#define sph_dec64le yerbas_cuda_dec64le
#define sph_dec64le_aligned yerbas_cuda_dec64le
#define sph_dec64be yerbas_cuda_dec64be
#define sph_dec64be_aligned yerbas_cuda_dec64be
#define sph_enc32le yerbas_cuda_enc32le
#define sph_enc32le_aligned yerbas_cuda_enc32le
#define sph_enc32be yerbas_cuda_enc32be
#define sph_enc32be_aligned yerbas_cuda_enc32be
#define sph_enc64le yerbas_cuda_enc64le
#define sph_enc64le_aligned yerbas_cuda_enc64le
#define sph_enc64be yerbas_cuda_enc64be
#define sph_enc64be_aligned yerbas_cuda_enc64be

'''


def load_helper(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"invalid inline include value: {spec}")
    include_name, helper_path = spec.split("=", 1)
    helper = pathlib.Path(helper_path).read_text(encoding="utf-8")
    helper = strip_c_linkage_wrappers(helper)
    helper = re.sub(r'(?m)^\s*#\s*include\s+[<\"].*[>\"]\s*$', '', helper)
    return include_name, helper


def inline_helpers(source: str, specs: list[str], all_specs: list[str]) -> str:
    # Normal helpers are replaced once. This matters for SHAvite, which has a
    # second aes_helper.c include inside a block comment for an alternate BE
    # implementation that must remain commented out.
    for spec in specs:
        include_name, helper = load_helper(spec)
        marker = re.compile(rf'(?m)^\s*#\s*include\s+"{re.escape(include_name)}"\s*$')
        if not marker.search(source):
            raise ValueError(f"include {include_name!r} not found in source")
        source = marker.sub(
            f"/* begin generated inline {include_name} */\n{helper}\n/* end generated inline {include_name} */",
            source,
            count=1,
        )

    # Some sphlib helpers are intentionally included several times with
    # different macro definitions (WHIRLPOOL includes md_helper.c three times).
    for spec in all_specs:
        include_name, helper = load_helper(spec)
        marker = re.compile(rf'(?m)^\s*#\s*include\s+"{re.escape(include_name)}"\s*$')
        if not marker.search(source):
            raise ValueError(f"include {include_name!r} not found in source")
        source = marker.sub(
            lambda _m: f"/* begin generated inline {include_name} */\n{helper}\n/* end generated inline {include_name} */",
            source,
        )
    return source


def transform_source(source: str, stem: str, inline_specs: list[str], inline_all_specs: list[str]) -> str:
    source = inline_helpers(source, inline_specs, inline_all_specs)
    source = strip_c_linkage_wrappers(source)
    source = re.sub(r'(?m)^\s*#\s*include\s+[<\"].*[>\"]\s*$', '', source)

    # File-scope helpers become device functions.
    source = re.sub(r'(?m)^static\s+', '__device__ static ', source)

    # Normal sphlib public entry points become private device functions.
    source = re.sub(
        rf'(?m)^void([ \t\r\n]+)(sph_{re.escape(stem)}[A-Za-z0-9_]*)',
        r'__device__ static void\1\2', source,
    )

    if stem == "whirlpool":
        # md_helper.c generates sph_HASH() through token-pasting, so there is
        # no literal sph_whirlpool name for the generic regex above to see.
        source = re.sub(
            r'(?m)^void\s*\n(SPH_XCAT\(sph_,\s*HASH\)\s*\()',
            r'__device__ static void\n\1',
            source,
        )
        # MAKE_CLOSE is another macro-generated public wrapper.
        source = re.sub(
            r'(?m)^void\s*\\\s*$',
            r'__device__ static void \\',
            source,
        )

    source = type_public_definitions(source, stem)

    # C permits implicit conversion from void*. C++/nvcc does not.
    source = re.sub(
        r'(?m)^(\s*)out\s*=\s*dst\s*;',
        r'\1out = static_cast<unsigned char *>(dst);', source,
    )
    source = re.sub(
        r'(?m)^(\s*)sc\s*=\s*cc\s*;',
        r'\1sc = static_cast<decltype(sc)>(cc);', source,
    )
    if stem == "simd":
        source = re.sub(r'\bd\s*=\s*dst\b', 'd = static_cast<unsigned char *>(dst)', source)

    # Echo, Hamsi and Shabal use byte-oriented internal core helpers while
    # their public sphlib API intentionally accepts const void*. C permits
    # that implicit conversion; C++/nvcc requires the byte pointer explicitly.
    if stem in ("echo", "hamsi", "shabal"):
        source = re.sub(
            r'\b(echo_small_core|echo_big_core|hamsi_small|hamsi_big|shabal_core)\(([^,\n]+),\s*data\s*,',
            r'\1(\2, static_cast<const unsigned char *>(data),',
            source,
        )

    names: list[str] = []
    for m in re.finditer(r'(?m)^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)', source):
        if m.group(1) not in names:
            names.append(m.group(1))
    if names:
        source += "\n\n/* generated macro cleanup */\n"
        source += "\n".join(f"#undef {name}" for name in reversed(names)) + "\n"

    source += "\n/* generated CUDA endian helper cleanup */\n"
    for name in (
        "sph_dec32le", "sph_dec32le_aligned", "sph_dec32be", "sph_dec32be_aligned",
        "sph_dec64le", "sph_dec64le_aligned", "sph_dec64be", "sph_dec64be_aligned",
        "sph_enc32le", "sph_enc32le_aligned", "sph_enc32be", "sph_enc32be_aligned",
        "sph_enc64le", "sph_enc64le_aligned", "sph_enc64be", "sph_enc64be_aligned",
    ):
        source += f"#undef {name}\n"
    return source


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--stem', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--header', required=True)
    p.add_argument('--out-source', required=True)
    p.add_argument('--out-header', required=True)
    p.add_argument('--inline-include', action='append', default=[])
    p.add_argument('--inline-include-all', action='append', default=[])
    a = p.parse_args()

    source = pathlib.Path(a.source).read_text(encoding='utf-8')
    header = pathlib.Path(a.header).read_text(encoding='utf-8')

    generated_header = (
        "/* Generated from pinned Yerbas Core sphlib source. Do not edit. */\n"
        "#include <cuda_runtime.h>\n#include <stddef.h>\n#include <stdint.h>\n#include <string.h>\n#include <limits.h>\n"
        + strip_public_prototypes(header, a.stem)
    )
    generated_source = (
        "/* Generated from pinned Yerbas Core sphlib source. Do not edit. */\n"
        + device_endian_helpers()
        + device_public_prototypes(header, a.stem)
        + transform_source(source, a.stem, a.inline_include, a.inline_include_all)
    )

    out_source = pathlib.Path(a.out_source)
    out_header = pathlib.Path(a.out_header)
    out_source.parent.mkdir(parents=True, exist_ok=True)
    out_header.parent.mkdir(parents=True, exist_ok=True)
    out_source.write_text(generated_source, encoding='utf-8')
    out_header.write_text(generated_header, encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
