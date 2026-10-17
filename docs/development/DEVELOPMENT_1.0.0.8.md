# Yerbas Miner 1.0.0.8 development build

1.0.0.7 remains the current stable release. This build is for validation.

## Windows dashboard

Automatic glyph selection now uses the existing broad-coverage block renderer in classic Windows Console Host. UTF-8 configuration alone does not make Braille or fractional-block glyphs available in the active console font. Windows Terminal retains the detailed renderer used on Linux. Explicit `--glyphs braille`, `--glyphs block` and `--glyphs ascii` selections take precedence. For Linux-style detail, use Windows Terminal with a font that covers the displayed glyphs.

Chart statistics now have explicit separators. Low/high statistics move to a second row when they would overrun the hashrate panel.

The Linux CUDA executable built successfully with native CPU compilation disabled and the existing broad CUDA architecture set. Existing correctness tests passed. Windows compilation and visual inspection are required before promoting this build. Automatic host selection is a compatibility heuristic; unsupported fonts can still require `--glyphs ascii`.

## Performance qualification

No new CPU or GPU performance kernel is enabled in this initial development build. An earlier scratchpad-load candidate passed reference equivalence but did not establish a repeatable stage-level improvement. Full-worker comparisons are evaluated separately before any optimization is promoted.

Configuration and tuning caches are preserved. Consensus, developer-fee behavior and stable-release assets are unchanged.
