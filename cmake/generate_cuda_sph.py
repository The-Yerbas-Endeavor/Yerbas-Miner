#!/usr/bin/env python3
"""Generate CUDA-device copies of selected sphlib hash implementations.

The source of truth remains the Yerbas Core revision pinned by CMake. This
performs a small mechanical transformation so the exact pinned sphlib code can
execute on CUDA without introducing a second algorithm implementation.
"""

from __future__ import annotations
import argparse
import pathlib
import re


def strip_public_prototypes(header: str, stem: str) -> str:
    # The implementation is included before its wrapper is called, so device
    # prototypes are unnecessary. Removing them also lets each implementation
    # live in a private C++ namespace without touching sphlib context types.
    pattern = re.compile(
        rf"(?ms)^\s*void\s+(sph_{re.escape(stem)}[A-Za-z0-9_]*)\s*\([^;]*?\);\s*"
    )
    return pattern.sub("", header)


def transform_source(source: str, stem: str) -> str:
    # Runtime/type headers are included globally by the generated type header;
    # do not include C/C++ standard headers from inside the private namespace.
    source = re.sub(r'(?m)^\s*#\s*include\s+[<\"].*[>\"]\s*$', '', source)

    # sphlib file-scope helpers/tables are at column zero. Device-decorating
    # only those declarations avoids rewriting normal local variables.
    source = re.sub(r'(?m)^static\s+', '__device__ static ', source)

    # Public sphlib entry points become private device functions. This handles
    # both one-line and two-line function declaration styles used by sphlib.
    source = re.sub(
        rf'(?m)^void([ \t\r\n]+)(sph_{re.escape(stem)}[A-Za-z0-9_]*)',
        r'__device__ static void\1\2',
        source,
    )

    # Prevent legacy implementation macros from leaking into the next hash
    # implementation included by stage_dispatch.cuh.
    names: list[str] = []
    for m in re.finditer(r'(?m)^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)', source):
        if m.group(1) not in names:
            names.append(m.group(1))
    if names:
        source += "\n\n/* generated macro cleanup */\n"
        source += "\n".join(f"#undef {name}" for name in reversed(names)) + "\n"
    return source


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--stem', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--header', required=True)
    p.add_argument('--out-source', required=True)
    p.add_argument('--out-header', required=True)
    a = p.parse_args()

    source = pathlib.Path(a.source).read_text(encoding='utf-8')
    header = pathlib.Path(a.header).read_text(encoding='utf-8')

    generated_header = (
        "/* Generated from pinned Yerbas Core sphlib source. Do not edit. */\n"
        "#include <cuda_runtime.h>\n"
        "#include <stddef.h>\n"
        "#include <stdint.h>\n"
        "#include <string.h>\n"
        "#include <limits.h>\n"
        + strip_public_prototypes(header, a.stem)
    )
    generated_source = (
        "/* Generated from pinned Yerbas Core sphlib source. Do not edit. */\n"
        + transform_source(source, a.stem)
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
