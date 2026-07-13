# N+1 first/second calibration-boundary audit

- Evaluated 200 paired health seeds without future demand.
- When one stack reaches 9.353029%, it is removed from service and the remaining two continue until a second stack reaches the same boundary.
- This boundary is the endpoint of the LZW calibration trajectory, not a physical failure threshold or a claimed system lifetime.

## Summary

| policy              |   runs |   first_boundary_mean_h |   second_boundary_mean_h |   second_boundary_q10_h |   second_boundary_median_h |   second_boundary_q90_h |   post_first_reserve_mean_h |   first_boundary_total_damage_mean_pct |   start_count_mean |   assignment_change_mean |
|:--------------------|-------:|------------------------:|-------------------------:|------------------------:|---------------------------:|------------------------:|----------------------------:|---------------------------------------:|-------------------:|-------------------------:|
| fixed_pair          |    200 |                1069.94  |                  1396.45 |                  1272.8 |                     1394   |                  1523.4 |                     326.515 |                                23.5423 |              3     |                    1     |
| order_blend_50      |    200 |                1144.58  |                  1401.24 |                  1272.8 |                     1402   |                  1528.4 |                     256.665 |                                24.3634 |              4.635 |                    2.635 |
| expected_n_plus_one |    200 |                 341.895 |                  1398.46 |                  1270.6 |                     1403.5 |                  1523.1 |                    1056.57  |                                15.8733 |              3     |                    1     |

## Paired against fixed pair

| policy              | reference   |   first_boundary_gain_mean_h |   second_boundary_gain_mean_h |   second_boundary_gain_ci95_low_h |   second_boundary_gain_ci95_high_h |   second_boundary_gain_median_h |   second_boundary_wins |   second_boundary_ties |   second_boundary_losses |   second_boundary_sign_p_one_sided |   second_boundary_win_share |   second_boundary_nonworse_share |   first_boundary_total_damage_delta_mean_pct |
|:--------------------|:------------|-----------------------------:|------------------------------:|----------------------------------:|-----------------------------------:|--------------------------------:|-----------------------:|-----------------------:|-------------------------:|-----------------------------------:|----------------------------:|---------------------------------:|---------------------------------------------:|
| order_blend_50      | fixed_pair  |                       74.64  |                         4.79  |                          2.73     |                            7.95218 |                               0 |                     39 |                    147 |                       14 |                        0.000401165 |                       0.195 |                             0.93 |                                     0.821107 |
| expected_n_plus_one | fixed_pair  |                     -728.045 |                         2.005 |                         -0.811541 |                            4.84373 |                               0 |                     79 |                     57 |                       64 |                        0.120802    |                       0.395 |                             0.68 |                                    -7.66898  |
