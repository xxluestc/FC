# 小时级N+1调度开发筛查

本实验使用指定的开发模板库所产生的Instant快层动作暴露，决策仅使用开发模板平均暴露，未知的执行暴露按小时重采样；启动损伤只在慢层在线集合实际变化时计入。健康边界是LZW标定轨迹终点，不是已辨识失效阈值，因此结果只用于筛选方法。

| policy        |   n_runs |   health_limit_crossing_share |   time_to_limit_mean_h |   time_to_limit_std_h |   time_to_limit_q10_h |   time_to_limit_median_h |   time_to_limit_q90_h |   start_count_mean |   assignment_change_mean |
|:--------------|---------:|------------------------------:|-----------------------:|----------------------:|----------------------:|-------------------------:|----------------------:|-------------------:|-------------------------:|
| fixed_pair    |       10 |                             1 |                 1331   |               6.69992 |                1325.7 |                   1328.5 |                1342   |                2   |                      0   |
| health_greedy |       10 |                             1 |                 1715.5 |              12.4655  |                1699.9 |                   1716.5 |                1731.1 |               35.7 |                     33.7 |

## 相对固定双堆的配对结果

| policy        | reference_policy   |   mean_gain_h |   minimum_gain_h |   maximum_gain_h |   win_share |   nonworse_share |
|:--------------|:-------------------|--------------:|-----------------:|-----------------:|------------:|-----------------:|
| health_greedy | fixed_pair         |         384.5 |              370 |              392 |           1 |                1 |

## 相对当前健康贪心的配对结果

| policy     | reference_policy   |   mean_gain_h |   minimum_gain_h |   maximum_gain_h |   win_share |   nonworse_share |
|:-----------|:-------------------|--------------:|-----------------:|-----------------:|------------:|-----------------:|
| fixed_pair | health_greedy      |        -384.5 |             -392 |             -370 |           0 |                0 |
