# 21UBE0022 identity and power audit

## Scope

The recent-year archive was scanned read-only and was not concatenated with the seven-day development canonical dataset. All power statistics below use rows with DC/DC input current at least 5 A.

## Identity evidence

- 7/7 Liu half-month files have a recent-archive counterpart with identical timestamp, target-power, DC/DC input-voltage and input-current rows.
- The archive filename date minus the actual first timestamp date is [1]; file names must not be treated as the measurement date without reading the timestamp column.
- This proves that the current seven-day load source belongs to vehicle `21UBE0022_苏E02625F` and is contained in the recent archive at the level of the four control-critical signals.
- It does not prove byte identity for every telemetry field, and it does not prove that the stack installed after April 2025 is the same physical stack as Liu's 170-cell 2022--2024 ageing stack.

## Empirical power evidence

| Signal | p99 (kW) | p99.9 (kW) | observed max (kW) |
|---|---:|---:|---:|
| Target command | 40.000 | 40.000 | 45.000 |
| Stack-side DC/DC input | 39.315 | 41.125 | 47.502 |
| Bus-side DC/DC output | 33.835 | 35.435 | 41.857 |
These values establish an empirical operating envelope and repeated controller commands, not a nameplate rated net power. A physical rating still requires a controller calibration table, vehicle specification or nameplate.

The source field `可加载功率` is excluded from the power table because its observed values reach 3000 and no unit definition is present in the available files. It is retained only as `loadable_raw` in machine-readable diagnostics.

## Health-observation consequence

- Headers declare 85 numbered voltage channels; this is not evidence of 85 physical PEMFC cells.
- On 6,129,950 running rows, the source's reported mean-voltage field agrees numerically with stack voltage divided by 85 declared channels within 0.06 V for 5,773,055 rows (94.18%). This is a telemetry-consistency check, not a physical cell-count inference.
- Across one sampled file per actual month, the median ratio of stack voltage to reported channel mean is 85.03. The modal number of nonzero numbered channels changes from 8 to 85, proving that channel availability/format changes within the archive.
- If the reported 1.3--1.5 V field is a channel-level mean, the stack/channel ratio near 85 is compatible with two-cell grouped channels and a 170-cell stack. If it is intended as a physical single-cell mean, its scale is inconsistent with normal loaded PEMFC voltage. Because the CSVs contain neither a channel definition nor a stack-ID field, physical cell count and stack continuity remain unproven.
- A separate 21UBE0022 voltage-trend measurement model may use total stack voltage after current/temperature matching and explicit telemetry-epoch filtering. It must not row-wise mix these records with the older MAT chain.
- The MAT chain remains a cross-dataset degradation prior. The recent archive is a candidate source for vehicle-specific correction and independent validation, not proof that both sources measured the same physical stack.
