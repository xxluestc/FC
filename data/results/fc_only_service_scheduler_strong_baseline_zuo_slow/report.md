# 小时级N+1调度开发筛查

本实验使用指定的开发模板库所产生的Instant快层动作暴露，决策仅使用开发模板平均暴露，未知的执行暴露按小时重采样；启动损伤只在慢层在线集合实际变化时计入。健康边界是LZW标定轨迹终点，不是已辨识失效阈值，因此结果只用于筛选方法。

| policy        |   n_runs |   health_limit_crossing_share |   time_to_limit_mean_h |   time_to_limit_std_h |   time_to_limit_q10_h |   time_to_limit_median_h |   time_to_limit_q90_h |   start_count_mean |   assignment_change_mean |
|:--------------|---------:|------------------------------:|-----------------------:|----------------------:|----------------------:|-------------------------:|----------------------:|-------------------:|-------------------------:|
| expected_max  |       20 |                             1 |                 1977.8 |               221.455 |                1696.4 |                   1946   |                2247.5 |              24.8  |                    22.8  |
| fixed_pair    |       20 |                             1 |                 1573.6 |               208.461 |                1303.4 |                   1608.5 |                1826.6 |               2    |                     0    |
| health_greedy |       20 |                             1 |                 1975.2 |               227.083 |                1688.1 |                   1949.5 |                2254   |              25.85 |                    23.85 |

## 相对固定双堆的配对结果

| policy        | reference_policy   |   mean_gain_h |   minimum_gain_h |   maximum_gain_h |   win_share |   nonworse_share |
|:--------------|:-------------------|--------------:|-----------------:|-----------------:|------------:|-----------------:|
| health_greedy | fixed_pair         |         401.6 |              121 |              592 |           1 |                1 |
| expected_max  | fixed_pair         |         404.2 |              196 |              594 |           1 |                1 |

## 相对当前健康贪心的配对结果

| policy       | reference_policy   |   mean_gain_h |   minimum_gain_h |   maximum_gain_h |   win_share |   nonworse_share |
|:-------------|:-------------------|--------------:|-----------------:|-----------------:|------------:|-----------------:|
| fixed_pair   | health_greedy      |        -401.6 |             -592 |             -121 |         0   |             0    |
| expected_max | health_greedy      |           2.6 |               -7 |               80 |         0.2 |             0.65 |
