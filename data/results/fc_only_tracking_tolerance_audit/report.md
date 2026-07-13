# 跟踪容差最坏案例边界审计

冻结30 kW主回放的最大误差案例为segment 42、`oldest_stack_0`、固定双堆、分配
`(1, 0)`。静态健康动作网格最大误差曾为4.018 kW；本审计保持真实段、在线健康和驻留
演化不变，只改变硬跟踪容差。

|   tracking_tolerance_kw | success   |   n_steps |   tracking_max_abs_kw |   tracking_mae_kw |   safety_override_steps |   expected_damage_increment_pct | error                                                   |
|------------------------:|:----------|----------:|----------------------:|------------------:|------------------------:|--------------------------------:|:--------------------------------------------------------|
|                   4     | False     |         0 |             nan       |         nan       |                     nan |                     nan         | no feasible multi-stack action for the requested demand |
|                   4.5   | False     |         0 |             nan       |         nan       |                     nan |                     nan         | no feasible multi-stack action for the requested demand |
|                   4.6   | False     |         0 |             nan       |         nan       |                     nan |                     nan         | no feasible multi-stack action for the requested demand |
|                   4.7   | False     |         0 |             nan       |         nan       |                     nan |                     nan         | no feasible multi-stack action for the requested demand |
|                   4.8   | False     |         0 |             nan       |         nan       |                     nan |                     nan         | no feasible multi-stack action for the requested demand |
|                   4.9   | False     |         0 |             nan       |         nan       |                     nan |                     nan         | no feasible multi-stack action for the requested demand |
|                   4.95  | True      |      4164 |               4.94112 |           1.81831 |                       9 |                       0.0129906 |                                                         |
|                   5     | True      |      4164 |               4.94112 |           1.81831 |                       9 |                       0.0129906 |                                                         |
|                   5.25  | True      |      4164 |               5.20743 |           1.81348 |                       9 |                       0.01299   |                                                         |
|                   5.4   | True      |      4164 |               5.29005 |           1.81434 |                       8 |                       0.0128686 |                                                         |
|                   5.45  | True      |      4164 |               5.29005 |           1.81434 |                       8 |                       0.0128686 |                                                         |
|                   5.49  | True      |      4164 |               5.48297 |           1.81454 |                       8 |                       0.0128682 |                                                         |
|                   5.499 | True      |      4164 |               5.49876 |           1.81474 |                       8 |                       0.0128678 |                                                         |
|                   5.5   | True      |      4164 |               5.49876 |           1.81474 |                       8 |                       0.0128678 |                                                         |

当前测试网格中的最小成功容差为4.95 kW。该结果说明5.5 kW附近的
容差来自离散动作、健康漂移和驻留共同作用，并非任意宽松常数；但它只审计冻结最坏案例，
不等同于所有留出案例在更紧容差下的成功率曲线。
