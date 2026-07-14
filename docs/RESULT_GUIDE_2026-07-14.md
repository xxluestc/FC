# 当前结果阅读指南

更新时间：2026-07-14

现在先看一张图：

- `data/results/figures/fc_only_foundation/fig32_simplified_random_load_comparison.png`

它回答本阶段最直接的问题：随机负载长什么样、DC-average如何轮换、五种策略的退化与氢耗位置、
以及Instant考虑健康前后到底改变了多少。图中四个分图依次是：

1. Zuo快变负载下DC-average的需求和三堆电流；
2. 各策略相对Average的最大单堆损伤变化；
3. 单位电量氢耗与最大单堆损伤的取舍；
4. Instant-health相对Instant-no-health的退化、氢耗和跟踪变化。

对应正式数据位于`data/results/simplified_random_load_comparison/`。优先看
`report.md`、`aggregate_metrics.csv`和`paired_health_ablation.csv`；逐运行表用于复核，不需要先读。

## 只在追溯问题时看

- `fig31_lzw_overall_exposure_screen.png`：说明为什么没有从LZW总体暴露硬拟合分动作退化率。这是负面
  诊断证据，不是策略效果图。
- `fig01_markov_timescale_audit.png`：说明实车经验负载和Zuo压力负载的时间尺度。
- `fig18_21ube0022_power_envelope.png`：说明40 kW为什么只是经验归一化参考，不是铭牌额定功率。
- `fig21_n_plus_one_service_boundary.png`、`fig23_n_plus_one_parameter_robustness.png`和
  `fig30_rate_bounded_blend_audit.png`：属于长期N+1研究，等短时基础链路理解清楚后再看。

`fig04`到`fig30`中的其余图保留为历史实验、边界审计或被否决方法的可追溯证据，不再作为当前入口。
本轮已删除冒烟测试目录和额外的`current_core`重复图目录；没有删除这些历史证据，以免后续重复走
已经证伪的路线。
