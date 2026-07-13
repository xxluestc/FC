# 小时级N+1调度开发筛查

本实验使用指定的开发模板库所产生的Instant快层动作暴露，决策仅使用开发模板平均暴露，未知的执行暴露按小时重采样；启动损伤只在慢层在线集合实际变化时计入。健康边界是LZW标定轨迹终点，不是已辨识失效阈值，因此结果只用于筛选方法。

| policy        |   n_runs |   health_limit_crossing_share |   time_to_limit_mean_h |   time_to_limit_std_h |   time_to_limit_q10_h |   time_to_limit_median_h |   time_to_limit_q90_h |   start_count_mean |   assignment_change_mean |
|:--------------|---------:|------------------------------:|-----------------------:|----------------------:|----------------------:|-------------------------:|----------------------:|-------------------:|-------------------------:|
| expected_max  |       20 |                             1 |                1682.4  |               179.046 |                1437.2 |                   1731.5 |                1875   |              24.8  |                    22.8  |
| fixed_pair    |       20 |                             1 |                1287.45 |               163.338 |                1054.7 |                   1320   |                1487.7 |               2    |                     0    |
| gamma_cvar    |       20 |                             1 |                1646.1  |               170.47  |                1406.6 |                   1668   |                1830.2 |              23.05 |                    21.05 |
| health_greedy |       20 |                             1 |                1680.8  |               179.935 |                1437.7 |                   1734   |                1875   |              25.7  |                    23.7  |

## 相对固定双堆的配对结果

| policy        | reference_policy   |   mean_gain_h |   minimum_gain_h |   maximum_gain_h |   win_share |   nonworse_share |
|:--------------|:-------------------|--------------:|-----------------:|-----------------:|------------:|-----------------:|
| health_greedy | fixed_pair         |        393.35 |              189 |              552 |           1 |                1 |
| expected_max  | fixed_pair         |        394.95 |              191 |              552 |           1 |                1 |
| gamma_cvar    | fixed_pair         |        358.65 |              187 |              493 |           1 |                1 |

## 相对当前健康贪心的配对结果

| policy       | reference_policy   |   mean_gain_h |   minimum_gain_h |   maximum_gain_h |   win_share |   nonworse_share |
|:-------------|:-------------------|--------------:|-----------------:|-----------------:|------------:|-----------------:|
| fixed_pair   | health_greedy      |       -393.35 |             -552 |             -189 |         0   |              0   |
| expected_max | health_greedy      |          1.6  |               -5 |               24 |         0.3 |              0.8 |
| gamma_cvar   | health_greedy      |        -34.7  |             -120 |               53 |         0.1 |              0.1 |
