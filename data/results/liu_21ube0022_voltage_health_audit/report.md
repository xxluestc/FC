# 21UBE0022 voltage-health signal audit

## Data treatment

- The archive remains separate from the older LZW MAT health chain.
- Files are assigned by the timestamp column, not by the one-day-shifted filename.
- Total stack voltage is matched at 90/120/160/195/270/340 A, actual-current tolerance 2.0 A, target-current tolerance 5.0 A, and coolant-in temperature 65.0--70.0 C.
- 2,635,767 of 7,684,841 rows pass these conditions (34.30%); 942 daily current-point records remain.
- The longitudinal composite uses only current points with at least 10,000 matched rows and 80% month coverage: 195 A, 270 A, 340 A. Sparse points remain in the daily/monthly tables but do not affect discontinuity or slope estimates.

## Discontinuity audit

The cross-current composite is centered within each current point. A discontinuity is declared only when the month-to-month shift exceeds max(0.8%, four robust MAD scales), here 3.998%.

- 2025-06: month-to-month composite shift +13.045%.
- 2025-08: month-to-month composite shift +4.921%.

These jumps are not interpreted as degradation. They may represent telemetry precision changes, controller calibration, maintenance or stack replacement. Without a stack-ID/maintenance log, the full archive cannot be treated as one uninterrupted ageing trajectory.

## Within-epoch trends

- Epoch 2 (2025-08 to 2026-06, 11 months): -1.984% per 100 days (95% CI -2.732 to -1.064).

After separate robust adjustment at each retained current point for actual current, coolant-in/out temperature, air-in pressure and hydrogen-in pressure, the final-epoch trend is -0.182% per 100 days (95% CI -0.639 to 0.060). The adjusted interval includes zero, so the apparent raw decline is not yet identifiable as irreversible ageing rather than changing operating conditions.

Within-epoch voltage shift is an observable performance residual, not true SOH and not the older MAT parameter trajectory. It may be monitored after execution, but it must not be converted directly into the world model's cumulative degradation state until a stack-specific measurement model or maintenance/stack-replacement log resolves this confounding. It is not suitable for identifying a universal Gamma degradation rate from the full recent-year archive.
