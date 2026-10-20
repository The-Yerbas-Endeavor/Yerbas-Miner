# Dashboard and scratchpad updates

The dashboard displays one stacked total hash-rate chart with a worker color key, worker contribution percentages and rolling averages. The chart uses a zero baseline and preserves space for activity, run time and rotation details. Worker colors support arbitrary device IDs and configurable palettes. See DASHBOARD_PALETTES.md and POWER_EFFICIENCY.md.

Linux fallback CPU scratchpad mappings reserve and trim an aligned range before requesting transparent huge pages. Explicit huge pages remain preferred; ordinary mapping and allocation fallbacks remain available. Other platforms retain their allocation paths. `YERBAS_CPU_ALIGN_SCRATCHPAD=0` selects the original Linux fallback mapping. No system huge-page configuration is changed.

The page-mode diagnostic reports transparent-hugepage-advised because an accepted advice request does not prove physical huge-page backing. Bookkeeping is thread-local. Hash arithmetic, runtime CPU dispatch and generic build settings are retained.
