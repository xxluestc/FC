# Rate-bounded Blend replay audit

- The rate-ratio limit is `1.10`, taken from the predeclared
  nominal factors `(1.00, 1.05, 1.10)` rather than tuned on these outcomes.
- Existing paired simulations are replayed; no trajectories are regenerated.

## Parameter robustness

| policy             |   scenarios |   blend_enabled_scenarios |   negative_n_plus_one_ci_scenarios |   positive_first_gain_scenarios |   minimum_n_plus_one_mean_h |
|:-------------------|------------:|--------------------------:|-----------------------------------:|--------------------------------:|----------------------------:|
| guarded_blend      |          41 |                       nan |                                  0 |                              15 |                        -0.5 |
| rate_bounded_blend |          41 |                        12 |                                  0 |                              12 |                        -0.5 |

## Cross-month summary

| scenario_id                          | policy             |   samples |   first_boundary_gain_mean_h |   first_boundary_gain_ci95_low_h |   first_boundary_gain_ci95_high_h |   second_boundary_gain_mean_h |   second_boundary_gain_ci95_low_h |   second_boundary_gain_ci95_high_h |   second_boundary_nonworse_share |   start_count_delta_mean |
|:-------------------------------------|:-------------------|----------:|-----------------------------:|---------------------------------:|----------------------------------:|------------------------------:|----------------------------------:|-----------------------------------:|---------------------------------:|-------------------------:|
| reference                            | guarded_blend      |        13 |                      71.0692 |                          46.0423 |                          100.107  |                       4.07692 |                           1.63533 |                            7.60892 |                         0.911538 |                  1.78462 |
| heterogeneity_gp_re_increased_perm_2 | guarded_blend      |        13 |                      50.7731 |                          41.638  |                           64.5138 |                       1.16154 |                          -2.02692 |                            5.06538 |                         0.7      |                  3.84615 |
| heterogeneity_gp_re_increased_perm_3 | guarded_blend      |        13 |                       0      |                           0      |                            0      |                       0       |                           0       |                            0       |                         1        |                  0       |
| reference                            | rate_bounded_blend |        13 |                      71.0692 |                          46.7036 |                          100.397  |                       4.07692 |                           1.61538 |                            7.43846 |                         0.911538 |                  1.78462 |
| heterogeneity_gp_re_increased_perm_2 | rate_bounded_blend |        13 |                       0      |                           0      |                            0      |                       0       |                           0       |                            0       |                         1        |                  0       |
| heterogeneity_gp_re_increased_perm_3 | rate_bounded_blend |        13 |                       0      |                           0      |                            0      |                       0       |                           0       |                            0       |                         1        |                  0       |
