# Held-out cross-month N+1 validation

- Frozen Blend 0.50 was evaluated on 13 calendar months with 20 paired seeds per month.
- The scheduler used only the development-template mean; monthly future exposure was hidden.
- The boundary is the LZW calibration endpoint, not a physical failure threshold.

## Primary calendar-month statistics

| policy              |   months |   first_boundary_gain_mean_h |   first_boundary_gain_ci95_low_h |   first_boundary_gain_ci95_high_h |   second_boundary_gain_mean_h |   second_boundary_gain_ci95_low_h |   second_boundary_gain_ci95_high_h |   second_boundary_gain_median_h |   second_boundary_better_months |   second_boundary_tied_months |   second_boundary_worse_months |   second_boundary_sign_p_one_sided |   seed_level_nonworse_share |
|:--------------------|---------:|-----------------------------:|---------------------------------:|----------------------------------:|------------------------------:|----------------------------------:|-----------------------------------:|--------------------------------:|--------------------------------:|------------------------------:|-------------------------------:|-----------------------------------:|----------------------------:|
| order_blend_50      |       13 |                      71.0692 |                          46.2539 |                           100.305 |                       4.07692 |                           1.60385 |                            7.43846 |                             0.2 |                               9 |                             1 |                              3 |                          0.072998  |                    0.911538 |
| expected_n_plus_one |       13 |                    -563.819  |                        -653.947  |                          -499.25  |                       3.41538 |                           1.16154 |                            6.67359 |                             3.3 |                              11 |                             0 |                              2 |                          0.0112305 |                    0.642308 |
