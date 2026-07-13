# Voltage-loss boundary mapping audit

- The LZW calibration endpoint is `9.353029%` damage.
- At that endpoint, fixed-condition voltage loss ranges from
  `0.883%` to
  `3.238%`
  over the audited current points.
- Every 5%, 10%, and 15% voltage-loss threshold lies outside the observed LZW
  theta trajectory. These mappings are extrapolation stress scenarios, not
  identified physical EOL thresholds.

## Selected mappings

|   current_a |   healthy_cell_voltage_v |   calibration_endpoint_cell_voltage_v |   calibration_endpoint_voltage_loss_fraction |   target_voltage_loss_fraction |   inferred_damage_boundary_pct |   damage_boundary_over_calibration | within_lzw_calibration_range   |    theta_i0 |   theta_ih |   theta_R_ohm |
|------------:|-------------------------:|--------------------------------------:|---------------------------------------------:|-------------------------------:|-------------------------------:|-----------------------------------:|:-------------------------------|------------:|-----------:|--------------:|
|         195 |                 0.742228 |                              0.729554 |                                    0.0170752 |                           0.05 |                        17.7163 |                            1.89418 | False                          | 1.52622e-07 | 0.00466511 |     0.104176  |
|         195 |                 0.742228 |                              0.729554 |                                    0.0170752 |                           0.1  |                        26.1814 |                            2.79924 | False                          | 1.05925e-07 | 0.00555928 |     0.159545  |
|         370 |                 0.695159 |                              0.672653 |                                    0.0323763 |                           0.05 |                        12.1928 |                            1.30362 | False                          | 1.72994e-07 | 0.00402074 |     0.0755157 |
|         370 |                 0.695159 |                              0.672653 |                                    0.0323763 |                           0.1  |                        18.5263 |                            1.98078 | False                          | 1.48969e-07 | 0.00475484 |     0.108902  |
