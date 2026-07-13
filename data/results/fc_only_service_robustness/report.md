# 扩展实车服务模型稳健性

Health-greedy与固定双堆在相同负载种子下配对；使用确定性条件均值执行，每次只改变一个物理口径。阴影范围为10个负载种子的最小至最大配对增益。

| policy        | reference_policy   |   mean_gain_h |   minimum_gain_h |   maximum_gain_h |   win_share |   nonworse_share | case                    | setting           |   multiplier |
|:--------------|:-------------------|--------------:|-----------------:|-----------------:|------------:|-----------------:|:------------------------|:------------------|-------------:|
| health_greedy | fixed_pair         |         407   |              400 |              418 |           1 |                1 | assignment_start_0p5    | assignment_start  |     0.5      |
| health_greedy | fixed_pair         |         384.5 |              370 |              392 |           1 |                1 | assignment_start_2      | assignment_start  |     2        |
| health_greedy | fixed_pair         |         400.7 |              397 |              408 |           1 |                1 | base                    | base              |     1        |
| health_greedy | fixed_pair         |         521.9 |              510 |              534 |           1 |                1 | continuous_0p5          | continuous        |     0.5      |
| health_greedy | fixed_pair         |         270.2 |              259 |              276 |           1 |                1 | continuous_2            | continuous        |     2        |
| health_greedy | fixed_pair         |         317.3 |              306 |              328 |           1 |                1 | health_limit_0p8        | health_limit      |     0.8      |
| health_greedy | fixed_pair         |         479.5 |              469 |              488 |           1 |                1 | health_limit_1p2        | health_limit      |     1.2      |
| health_greedy | fixed_pair         |         436.9 |              426 |              450 |           1 |                1 | load_shift_0p5          | load_shift        |     0.5      |
| health_greedy | fixed_pair         |         339.8 |              329 |              356 |           1 |                1 | load_shift_2            | load_shift        |     2        |
| health_greedy | fixed_pair         |         491.5 |              484 |              498 |           1 |                1 | operational_start_0p434 | operational_start |     0.434192 |
| health_greedy | fixed_pair         |         266.6 |              263 |              274 |           1 |                1 | operational_start_2p396 | operational_start |     2.39571  |

全扫描最小配对增益为259.00 h，最低获胜率为100.0%。
