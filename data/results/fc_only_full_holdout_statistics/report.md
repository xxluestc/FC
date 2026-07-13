# 完整留出 segment 级不确定性

重采样单位为 8 个完整正功率 segment，不把逐秒样本视为独立观测。两个非平凡健康身份是预声明的堆身份循环，分别检验 health-greedy 相对固定双堆的终端最大退化差是否小于 0；Holm 校正控制两个主检验的多重性。

| health_case    |   n_segments | metric                      |    estimate |   bootstrap_mean |   ci95_lower |   ci95_upper |
|:---------------|-------------:|:----------------------------|------------:|-----------------:|-------------:|-------------:|
| oldest_stack_0 |            8 | terminal_max_delta_mean_pct | -0.00961916 |      -0.00961153 |   -0.014611  |  -0.0052627  |
| oldest_stack_1 |            8 | terminal_max_delta_mean_pct | -0.00993091 |      -0.00995269 |   -0.0152127 |  -0.00539473 |

| health_case    | hypothesis                                        |   n_segments |   wilcoxon_statistic |   p_value_one_sided |   better_segments |   nonworse_segments |   p_value_holm | reject_holm_0p05   |
|:---------------|:--------------------------------------------------|-------------:|---------------------:|--------------------:|------------------:|--------------------:|---------------:|:-------------------|
| oldest_stack_0 | health_greedy_minus_fixed_terminal_max_damage < 0 |            8 |                    0 |          0.00390625 |                 8 |                   8 |      0.0078125 | True               |
| oldest_stack_1 | health_greedy_minus_fixed_terminal_max_damage < 0 |            8 |                    0 |          0.00390625 |                 8 |                   8 |      0.0078125 | True               |
