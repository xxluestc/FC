# 小时级N+1调度开发筛查

本实验使用指定的开发模板库所产生的Instant快层动作暴露，决策仅使用开发模板平均暴露，未知的执行暴露按小时重采样；启动损伤只在慢层在线集合实际变化时计入。健康边界是LZW标定轨迹终点，不是已辨识失效阈值，因此结果只用于筛选方法。

| policy        |   n_runs |   health_limit_crossing_share |   time_to_limit_mean_h |   time_to_limit_std_h |   time_to_limit_q10_h |   time_to_limit_median_h |   time_to_limit_q90_h |   start_count_mean |   assignment_change_mean |
|:--------------|---------:|------------------------------:|-----------------------:|----------------------:|----------------------:|-------------------------:|----------------------:|-------------------:|-------------------------:|
| expected_max  |       20 |                             1 |                1961.65 |               216.333 |                1687.4 |                   1943   |                2222.2 |              23.1  |                    21.1  |
| fixed_pair    |       20 |                             1 |                1583.35 |               213.202 |                1299.8 |                   1629.5 |                1825.4 |               2    |                     0    |
| health_greedy |       20 |                             1 |                1960.15 |               216.225 |                1687.5 |                   1943   |                2219.5 |              25.65 |                    23.65 |

## 相对固定双堆的配对结果

| policy        | reference_policy   |   mean_gain_h |   minimum_gain_h |   maximum_gain_h |   win_share |   nonworse_share |
|:--------------|:-------------------|--------------:|-----------------:|-----------------:|------------:|-----------------:|
| health_greedy | fixed_pair         |         376.8 |               96 |              545 |           1 |                1 |
| expected_max  | fixed_pair         |         378.3 |               96 |              559 |           1 |                1 |

## 相对当前健康贪心的配对结果

| policy       | reference_policy   |   mean_gain_h |   minimum_gain_h |   maximum_gain_h |   win_share |   nonworse_share |
|:-------------|:-------------------|--------------:|-----------------:|-----------------:|------------:|-----------------:|
| fixed_pair   | health_greedy      |        -376.8 |             -545 |              -96 |        0    |              0   |
| expected_max | health_greedy      |           1.5 |              -25 |               40 |        0.25 |              0.8 |
