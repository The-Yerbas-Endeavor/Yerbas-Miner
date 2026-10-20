# Power efficiency display

Worker telemetry rows include EFF(H/s/W), current hash rate divided by reported watts, and ENERGY(J/kH), reported watts divided by hash rate and multiplied by1000. Higher H/s/W and lower J/kH indicate better efficiency for the observed workload.

The CURRENT HASHRATE row includes combined CPU+GPU POWER, EFF and ENERGY. Combined metrics require valid readings for every enabled CPU/GPU worker. Missing, nonfinite or nonpositive power readings produce n/a; idle workers with valid power show0H/s/W and n/aJ/kH. No partial-power denominator is paired with the full miner hash rate.

These are estimates from the existing CPU/GPU telemetry sample, refreshed approximately every10seconds, and the current hash reporting window. They describe reported component power, not total system wall power or mining-only incremental power. They include other work running on those components. Changing GhostRider rotations also changes instantaneous efficiency. No new sensor polling, pool calls or timing tests are introduced.

The additional fields reuse existing rows, preserving Activity space. Calculations are generic for any number of workers and do not depend on hardware models.
