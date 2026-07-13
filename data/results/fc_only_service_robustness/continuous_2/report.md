# 小时级N+1调度开发筛查

本实验使用指定的开发模板库所产生的Instant快层动作暴露，决策仅使用开发模板平均暴露，未知的执行暴露按小时重采样；启动损伤只在慢层在线集合实际变化时计入。健康边界是LZW标定轨迹终点，不是已辨识失效阈值，因此结果只用于筛选方法。

| policy        |   n_runs |   health_limit_crossing_share |   time_to_limit_mean_h |   time_to_limit_std_h |   time_to_limit_q10_h |   time_to_limit_median_h |   time_to_limit_q90_h |   start_count_mean |   assignment_change_mean |
|:--------------|---------:|------------------------------:|-----------------------:|----------------------:|----------------------:|-------------------------:|----------------------:|-------------------:|-------------------------:|
| fixed_pair    |       10 |                             1 |                  894.6 |               4.94862 |                 889   |                      893 |                 901.2 |                2   |                      0   |
| health_greedy |       10 |                             1 |                 1164.8 |               7.85706 |                1151.7 |                     1167 |                1171.1 |               24.8 |                     22.8 |

## 相对固定双堆的配对结果

| policy        | reference_policy   |   mean_gain_h |   minimum_gain_h |   maximum_gain_h |   win_share |   nonworse_share |
|:--------------|:-------------------|--------------:|-----------------:|-----------------:|------------:|-----------------:|
| health_greedy | fixed_pair         |         270.2 |              259 |              276 |           1 |                1 |

## 相对当前健康贪心的配对结果

| policy     | reference_policy   |   mean_gain_h |   minimum_gain_h |   maximum_gain_h |   win_share |   nonworse_share |
|:-----------|:-------------------|--------------:|-----------------:|-----------------:|------------:|-----------------:|
| fixed_pair | health_greedy      |        -270.2 |             -276 |             -259 |           0 |                0 |
