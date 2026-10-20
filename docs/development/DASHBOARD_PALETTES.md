# Dashboard worker palettes

The default `blue` palette starts with deep blue, cyan and ice blue. CPU owns color slot0; a GPU uses its device ID plus1. Each stack band and its worker-key label use the same foreground/background pair. GPU count and hardware model do not select the palette. Sparse or reordered device IDs retain their colors.

Select a built-in palette with `YERBAS_CHART_PALETTE=blue`, `green`, `ocean`, or `neutral`. The value is read at startup; unknown names use blue. Example on Linux:

```sh
YERBAS_CHART_PALETTE=blue ./yerbas-miner
```

On PowerShell:

```powershell
$env:YERBAS_CHART_PALETTE="blue"
.\yerbas-miner.exe
```

`YERBAS_CHART_COLORS` optionally supplies comma-separated ANSI256 color indices in CPU,GPU0,GPU1,... slot order, for example `25,38,153,69,30,117`. Entries must be0–255. Duplicate indices are removed; malformed lists fall back to the selected built-in palette. Short custom lists extend automatically from the selected palette. The built-ins first extend their preferred family, then add remaining ANSI indices, offering256 distinct indices before repeating. Large worker counts remain supported; labels always identify workers, and sufficiently large terminals are needed to display long keys. Terminal palettes have finite color and cell resolution.

These settings change display colors only. Generic CPU dispatch, CUDA architecture coverage, hash calculations and tuning remain unchanged.
