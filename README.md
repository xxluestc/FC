# FC: degradation-aware fuel-cell power management

Reproducible research code for the minimum route:

`vehicle speed/state -> short-horizon demand forecast -> vehicle dynamics -> degradation-aware MPC power allocation`.

The repository does not contain private raw vehicle data, large MATLAB files, theses, or credentials. Local data paths belong in `configs/paths.local.yaml` (copy from the template).

## Pipeline

1. `scripts/00_audit_data.py`: inspect Han/Liu/Li data sources and select a canonical chain.
2. `scripts/01_preprocess_data.py`: normalize time, units, missing values, and signals.
3. `scripts/02_build_stack_degradation_h2.py`: I-V-P, degradation proxy and hydrogen model.
4. `scripts/03_vehicle_dynamics_power.py`: speed-to-demand vehicle dynamics and calibration.
5. `scripts/04_train_or_run_predictors.py`: persistence, AR, speed/condition-aware predictors.
6. `scripts/05_run_power_allocation.py`: instant and receding-horizon allocation.
7. `scripts/06_run_all_experiments.py`: reproducible end-to-end experiment.

## Current reproducible result

- Canonical source: Liu 21UBE0022 half-month vehicle CSV (timestamp, speed, SOC, FC, battery and motor signals in one table).
- Dynamics test: MAE 8.55 kW, RMSE 17.85 kW, R² 0.877.
- State-aware prediction: 1 s speed RMSE 0.355 m/s; corrected demand MAE/RMSE 6.76/14.32 kW.
- Predicted MPC versus constant MPC: hydrogen -4.33%, FC degradation proxy -12.38%, switches -9.28%, FC variation -14.34%; terminal SOC maintained.

See `reports/` for scope and limitations. Raw data and locally generated large trajectories are intentionally excluded.

## Local execution

```powershell
python scripts/00_audit_data.py --han <han_dir> --liu <liu_dir> --li <li_dir> --out reports/data_audit_inventory.json
python scripts/01_preprocess_data.py --input-dir <liu_half_month_csv_dir> --output data/processed/liu_vehicle_canonical_1s.csv --summary data/processed/preprocess_summary.json
python scripts/03_vehicle_dynamics_power.py --input data/processed/liu_vehicle_canonical_1s.csv --output data/processed/power_demand_from_dynamics.csv --metrics data/results/vehicle_dynamics_metrics.json
python scripts/04_train_or_run_predictors.py --input data/processed/power_demand_from_dynamics.csv --dynamics-metrics data/results/vehicle_dynamics_metrics.json --output data/processed/prediction_results.csv --metrics data/results/prediction_metrics.csv
python scripts/06_run_all_experiments.py
python scripts/plot_results.py
```
