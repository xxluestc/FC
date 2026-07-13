# Project route

1. Audit Han, Liu, and Li datasets for timestamp, speed, power, SOC, current, and voltage.
2. Select a canonical speed-and-power source without mixing incompatible vehicles silently.
3. Reconstruct demand using traceable vehicle dynamics and measured-power calibration.
4. Predict speed/state first; convert predicted motion into demand power.
5. Build Liu-data-derived stack degradation and theoretical/calibrated hydrogen maps.
6. Compare instant, persistence MPC, predicted MPC, and perfect-preview MPC.

## Current source ownership

- Liu 21UBE0022: canonical vehicle motion and power signals.
- Liu `data_mark/x_est`: 2022--2024 170-cell degradation prior. It has no auditable row-wise link to the 2025--2026 21UBE0022 stream and is intentionally kept separate.
- 21UBE0022 recent archive: independent identity/power/voltage-epoch audit; 40 kW empirical operating reference, not a nameplate rating.
- Han 21UBE0025: vehicle parameters, kinematic-fragment clustering and Markov methodology; independent vehicle, not row-wise merged.
- Li engine datasets: multi-stack voltage/parameter-estimation reference; no vehicle speed.
