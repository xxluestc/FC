# Frozen N+1 parameter robustness

- 41 predeclared scenarios, 20 paired seeds per scenario.
- Literature random-effect quantiles are normalized stress cases, not identified vehicle-specific stack factors.
- Blend 0.50 is frozen and is not retuned in this audit.
- The health boundary is a calibration endpoint, not physical EOL.

## Robustness summary

| policy         |   scenarios |   positive_first_mean_scenarios |   nonnegative_second_mean_scenarios |   positive_second_ci_scenarios |   negative_second_ci_scenarios |   second_gain_mean_across_scenarios_h |   second_gain_min_scenario_h | second_gain_min_scenario_id          |   minimum_seed_nonworse_share |
|:---------------|------------:|--------------------------------:|------------------------------------:|-------------------------------:|-------------------------------:|--------------------------------------:|-----------------------------:|:-------------------------------------|------------------------------:|
| health_greedy  |          41 |                              41 |                                  26 |                             14 |                              7 |                              108.707  |                       -28.35 | heterogeneity_gp_re_increased_perm_3 |                          0.05 |
| order_blend_50 |          41 |                              31 |                                  33 |                             19 |                              1 |                              111.274  |                       -47.2  | heterogeneity_gp_re_increased_perm_3 |                          0.25 |
| guarded_blend  |          41 |                              15 |                                  39 |                              7 |                              0 |                               17.9207 |                        -0.5  | boundary_scale_1.20                  |                          0.65 |
