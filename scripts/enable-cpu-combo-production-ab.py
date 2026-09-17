#!/usr/bin/env python3
from pathlib import Path

path = Path("src/ghostrider/ghostrider.cpp")
text = path.read_text()
original = text

include_old = '#include "cpu/cn_2way.h"\n'
include_new = '#include "cpu/cn_2way.h"\n#include "cpu/cpu_combo_policy.h"\n'
if '#include "cpu/cpu_combo_policy.h"' not in text:
    if include_old not in text:
        raise SystemExit("ERROR: include anchor not found")
    text = text.replace(include_old, include_new, 1)

schedule_old = '    const auto& schedule = cached_schedule(works[0]);\n    uint512 hash[4][18]{};\n'
schedule_new = (
    '    const auto& schedule = cached_schedule(works[0]);\n'
    '    const auto effective_cn_widths =\n'
    '        ::yerbas::cpu_combo_policy::select(schedule, cn_widths);\n'
    '    uint512 hash[4][18]{};\n'
)
if 'const auto effective_cn_widths' not in text:
    if schedule_old not in text:
        raise SystemExit("ERROR: batch schedule anchor not found")
    text = text.replace(schedule_old, schedule_new, 1)

width_old = '        const unsigned int width = cn_widths[static_cast<std::size_t>(algorithm)];\n'
width_new = '        const unsigned int width = effective_cn_widths[static_cast<std::size_t>(algorithm)];\n'
if width_old in text:
    text = text.replace(width_old, width_new, 1)
elif width_new not in text:
    raise SystemExit("ERROR: CN width anchor not found")

if text == original:
    print("CPU combo production A/B patch already applied.")
else:
    path.write_text(text)
    print("Applied CPU combo production A/B patch to src/ghostrider/ghostrider.cpp")
