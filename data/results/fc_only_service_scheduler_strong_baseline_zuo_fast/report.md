# 小时级N+1调度开发筛查

本实验使用指定的开发模板库所产生的Instant快层动作暴露，决策仅使用开发模板平均暴露，未知的执行暴露按小时重采样；启动损伤只在慢层在线集合实际变化时计入。健康边界是LZW标定轨迹终点，不是已辨识失效阈值，因此结果只用于筛选方法。

| policy        |   n_runs |   health_limit_crossing_share |   time_to_limit_mean_h |   time_to_limit_std_h |   time_to_limit_q10_h |   time_to_limit_median_h |   time_to_limit_q90_h |   start_count_mean |   assignment_change_mean |
|:--------------|---------:|------------------------------:|-----------------------:|----------------------:|----------------------:|-------------------------:|----------------------:|-------------------:|-------------------------:|
| expected_max  |       20 |                             1 |                1075.85 |               98.4764 |                 953.9 |                   1076   |                1197.2 |              15.65 |                    13.65 |
| fixed_pair    |       20 |                             1 |                 840.7  |              111.469  |                 663.7 |                    845   |                 948   |               2    |                     0    |
| health_greedy |       20 |                             1 |                1072.15 |              109.852  |                 954.1 |                   1075.5 |                1198.8 |              16.75 |                    14.75 |

## 相对固定双堆的配对结果

| policy        | reference_policy   |   mean_gain_h |   minimum_gain_h |   maximum_gain_h |   win_share |   nonworse_share |
|:--------------|:-------------------|--------------:|-----------------:|-----------------:|------------:|-----------------:|
| health_greedy | fixed_pair         |        231.45 |              151 |              301 |           1 |                1 |
| expected_max  | fixed_pair         |        235.15 |              149 |              301 |           1 |                1 |

## 相对当前健康贪心的配对结果

| policy       | reference_policy   |   mean_gain_h |   minimum_gain_h |   maximum_gain_h |   win_share |   nonworse_share |
|:-------------|:-------------------|--------------:|-----------------:|-----------------:|------------:|-----------------:|
| fixed_pair   | health_greedy      |       -231.45 |             -301 |             -151 |        0    |              0   |
| expected_max | health_greedy      |          3.7  |              -29 |               83 |        0.25 |              0.8 |
