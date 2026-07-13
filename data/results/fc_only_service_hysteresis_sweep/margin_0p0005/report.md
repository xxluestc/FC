# 小时级N+1调度开发筛查

本实验使用指定的开发模板库所产生的Instant快层动作暴露，决策仅使用开发模板平均暴露，未知的执行暴露按小时重采样；启动损伤只在慢层在线集合实际变化时计入。健康边界是LZW标定轨迹终点，不是已辨识失效阈值，因此结果只用于筛选方法。

| policy              |   n_runs |   health_limit_crossing_share |   time_to_limit_mean_h |   time_to_limit_std_h |   time_to_limit_q10_h |   time_to_limit_median_h |   time_to_limit_q90_h |   start_count_mean |   assignment_change_mean |
|:--------------------|---------:|------------------------------:|-----------------------:|----------------------:|----------------------:|-------------------------:|----------------------:|-------------------:|-------------------------:|
| expected_hysteresis |       10 |                             1 |                 1733.5 |               6.3814  |                1723.8 |                   1734   |                1740   |               34.8 |                     32.8 |
| fixed_pair          |       10 |                             1 |                 1331.5 |               6.65415 |                1325.7 |                   1329.5 |                1342   |                2   |                      0   |
| health_greedy       |       10 |                             1 |                 1732.2 |               7.68548 |                1723.6 |                   1733.5 |                1740.1 |               36.2 |                     34.2 |

## 相对固定双堆的配对结果

| policy              | reference_policy   |   mean_gain_h |   minimum_gain_h |   maximum_gain_h |   win_share |   nonworse_share |
|:--------------------|:-------------------|--------------:|-----------------:|-----------------:|------------:|-----------------:|
| health_greedy       | fixed_pair         |         400.7 |              397 |              408 |           1 |                1 |
| expected_hysteresis | fixed_pair         |         402   |              396 |              408 |           1 |                1 |

## 相对当前健康贪心的配对结果

| policy              | reference_policy   |   mean_gain_h |   minimum_gain_h |   maximum_gain_h |   win_share |   nonworse_share |
|:--------------------|:-------------------|--------------:|-----------------:|-----------------:|------------:|-----------------:|
| fixed_pair          | health_greedy      |        -400.7 |             -408 |             -397 |         0   |              0   |
| expected_hysteresis | health_greedy      |           1.3 |               -2 |               11 |         0.2 |              0.5 |
