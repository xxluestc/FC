# 预测 MPC 与退化代理功率分配复验

## 搜索设置

采用离散档位 beam-search MPC。权重在连续测试段前 1,200 s 上做 24 点确定性结构网格搜索，中心为 `w_deg=2`，同时搜索氢耗、SOC、`w_switch`、`w_smooth={0.05,0.1,0.2}` 和预测置信衰减。候选若令 Predicted H5 或 Instant H1 的末端 SOC 偏差超过 ±0.02，会受到不可行惩罚。

最优校准参数为：`w_H2=0.45, w_deg=2.5, w_smooth=0.10, w_SOC=3.0, w_switch=0.10`，预测衰减 `alpha(h)=exp(-0.08h)`。完整搜索见 `data/results/allocation/mpc_weight_search.csv`。

## 完整 3,600 s 结果

| 策略 | H | 氢耗 (kg) | proxy 累计 | SOC末值 | 电池吞吐 (kWh) | 切换 | FC总变差 (kW) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Instant | 1 | 0.30113 | 162.07 | 0.69877 | 16.354 | 53 | 281.2 |
| Constant | 5 | 0.30208 | 182.50 | 0.69811 | 16.218 | 95 | 481.1 |
| Perfect | 5 | 0.30115 | 177.35 | 0.69810 | 16.057 | 78 | 409.4 |
| Predicted | 5 | 0.30080 | 174.54 | 0.69812 | 16.050 | 84 | 425.3 |
| Constant | 10 | 0.30984 | 206.76 | 0.70090 | 16.034 | 93 | 479.9 |
| Perfect | 10 | 0.30145 | 180.51 | 0.69809 | 16.039 | 70 | 351.5 |
| Predicted | 10 | 0.30086 | 175.58 | 0.69807 | 16.016 | 88 | 439.9 |

Predicted H5 相对 Constant H5：氢耗降低 0.423%，proxy 降低 4.365%，电池吞吐降低 1.034%，切换降低 11.579%，FC 总变差降低 11.610%。相对 Instant，Predicted H5 的氢耗降低 0.109%、电池吞吐降低 1.856%，但 proxy 高 7.692%、切换高 58.49%；因此不能声称全面优于 Instant。

Predicted 与 Perfect 的档位不同率为 H3 12.86%、H5 22.28%、H10 26.53%，随预测域增长而扩大，和 H10 预测误差恶化一致。

## 诊断文件

- `strategy_tier_occupancy.csv`：每种策略和 H 的档位数量/占比。
- `tier_proxy_contribution.csv`：各档位原始及加权 proxy 贡献。
- `braking_allocation_diagnostics.csv`：制动/非制动段 FC、电池、吞吐和 proxy。
- `predicted_vs_perfect_actions.csv`：逐秒动作档位差。
- `allocation_trajectory.csv`：逐秒氢耗、proxy、切换、平滑成本明细。

## 解释边界

这里的 degradation proxy 来自刘占伟老化参数/性能损失表，不是真实动作退化系数；其 `x_est` 与 21UBE0022 半月 CSV 尚未确认同车同堆。MPC 结果只能证明该代理进入目标函数后会改变档位决策，不能证明材料退化率已被准确辨识。
