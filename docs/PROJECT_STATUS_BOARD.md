# 项目进度总表

更新时间：2026-07-13

本表用于快速查看当前阶段、证据、下一步和论文图状态。研究口径与详细门槛仍以
`docs/MASTER_RESEARCH_PLAN.md`为准。

## 总体阶段

| 阶段 | 核心目标 | 状态 | 已有证据 | 下一出口 |
|---|---|---|---|---|
| G0 | 仓库、数据与交接可复现 | 已完成 | 数据物化、96项测试、baseline | 持续回归 |
| G1 | 三堆N+1确定性在线健康闭环 | 已完成 | 动作驱动D→theta→IV/功率；轮换测试 | 保持冻结 |
| G2 | FC-only内层功率接口 | 已完成 | 无电池/SOC依赖；跟踪约束 | 保持冻结 |
| G3 | 实车标定随机负载与Zuo压力场景 | 标定完成、容量口径待确认 | 实车1秒主基线；Zuo快/慢；留出容量审计 | 确认物理额定功率 |
| G4 | 确定性策略公平比较 | 已完成基础出口 | 10种子随机场景；24个真实留出窗口 | 保留全量回放扩展 |
| G5 | Gamma随机退化与配对统计 | 已完成时间尺度诊断 | 90次在线配对；短时Gamma不适用 | 转入小时级风险调度 |
| G6 | 实车健康观测校正 | 等待数据链 | 仅有动作驱动预测态 | MAT变量链与observer接口 |
| G7 | 双时间尺度调度与机制消融 | 冻结留出验证完成 | 144个完整段案例、518,490步、零违规 | 论文证据收束 |
| G8 | 锂电池外层、需求预测、高级算法 | 冻结 | 仅保留接口和适配判断 | 基础门槛完成后再启动 |

## 当前执行表

| 优先级 | 工作项 | 状态 | 验收标准 |
|---:|---|---|---|
| 1 | G3-G4关键结果论文图 | 已完成 | PNG 320 DPI、脚本、视觉检查、图注清单 |
| 2 | FC-only Gamma配对与时间尺度诊断 | 已完成 | 明确短时只用条件均值 |
| 3 | 事件退化系数消融 | 已完成首轮 | 75点、5种子、三场景、论文图 |
| 4 | 小时级N+1休息堆调度 | 已完成开发筛查 | health-greedy四场景20/20优于固定双堆 |
| 5 | 长期调度稳健性扩展 | 已完成开发出口 | 120模板；11组单因素全10/10正增益；强基线 |
| 6 | 冻结方法真实留出验证 | 已完成选择层 | 18/18可行；health-greedy oracle集合命中100% |
| 7 | 留出全量连续快层回放 | 已完成 | 86,415样本×3健康身份×2策略；零违规；最大误差5.499 kW |
| 8 | 负载归一化与N+1容量审计 | 诊断完成、物理确认待输入 | 30/35/40 kW全量敏感性；36.756 kW事后下界；40 kW候选不作额定值 |
| 9 | 论文证据与核心章节初稿 | 已完成首版 | 统一公式、主张矩阵、16张规范表、segment统计、核心章节初稿 |
| 10 | 真实健康观测变量审计 | 等待输入 | 明确可观测量、时间同步和噪声模型 |
| 11 | 严格内部审稿与修订 | 已完成首轮 | 三类审稿人、共识风险、可执行发表门槛 |
| 12 | 跟踪容差边界审计 | 已完成定向审计 | 最坏案例4.90 kW失败、4.95 kW成功；5.5 kW余量0.55 kW |
| 13 | 风险方法自身目标遗憾 | 已完成 | 12/12在线集合一致；三策略Expected/CVaR最大遗憾均为0 |
| 14 | 摘要与结论初稿 | 已完成 | 期刊中立中文稿；主结果、统计和限制均来自规范证据 |

## 关键判断

| 判断 | 当前结论 | 边界 |
|---|---|---|
| 主负载时间基准 | 实车1秒经验矩阵 | Zuo 30秒仅是压力场景工程假设 |
| 当前确定性健康基线 | Instant-health | Beam无稳定物理优势且约增加107%规划时间 |
| Average基线 | 保留对照 | 真实窗口中22/24成功，不能称普遍可行 |
| Rotating基线 | 保留公平性对照 | 轮休更均衡，但启停退化代价明显 |
| Gamma在线采样 | 不用于120秒策略均值排序 | shape约`1e-3`，100%运行连续采样近零 |
| Gamma不确定性 | 进入小时级风险/维护层 | 旧1000小时重复结果仅作历史诊断 |
| 当前方法证据 | 秒级Instant不足以单独成文 | 相对Average仅小幅且系数敏感 |
| 慢层主方法 | 24小时低频health-greedy | Expected-max仅比它高1.5-3.7 h且获胜率20%-30% |
| Gamma-CVaR | 降级为离线消融 | 四场景均未稳定超过确定性方法 |
| 滞回增强 | 不进入主方法 | 无阈值通过冻结的逐种子非劣规则 |
| 长期结果边界 | 仍是边界到达仿真 | 120个开发模板；LZW终点不是失效阈值 |
| 完整真实留出 | 冻结health-greedy方向有效且全例可执行 | 8段segment级统计成立；最长段7.543 h，只验证入口选择和段内快层；高于30 kW样本被截峰 |
| 负载容量口径 | 30 kW不是额定功率 | 30/35/40 kW截峰11.585%/2.672%/0%，C8方向均保持；最坏组合30 kW不截峰6.65%超容量；40 kW需外部确认 |
| 电池与未来预测 | 暂不进入当前优化 | 后续作为外层与独立优化点 |

## 图表计划

| 图号候选 | 内容 | 数据 | 状态 |
|---|---|---|---|
| Fig. 1 | Markov时间尺度与状态变化率 | `fc_only_load_sensitivity` | 已完成 |
| Fig. 2 | 10种子策略跟踪-氢耗-退化权衡 | `fc_only_deterministic_comparison` | 已完成 |
| Fig. 3 | 真实留出窗口成功率、跟踪与计算代价 | `fc_only_real_holdout_replay` | 已完成 |
| Fig. 4 | Gamma有效shape与近零概率 | `fc_only_gamma_timescale` | 已完成 |
| Fig. 5 | 旧1000小时聚合Gamma CV敏感性 | `fc_only_gamma_aggregate` | 降级为历史诊断 |
| Fig. 6 | 75点连续/启停/变载机制消融 | `fc_only_mechanism_ablation` | 已完成 |
| Fig. 7 | 小时级策略健康边界生存与区间 | `fc_only_service_scheduler` | 已完成开发图 |
| Fig. 8 | 6-48小时最小重调度周期敏感性 | `fc_only_service_scheduler_reschedule` | 已完成开发图 |
| Fig. 9 | Gamma CV 5%-20%长期敏感性 | `fc_only_service_scheduler_gamma_cv` | 已完成开发图 |
| Fig. 10 | 健康边界与四类机理单因素稳健性 | `fc_only_service_robustness` | 已完成 |
| Fig. 11 | 120模板暴露与四场景强基线结果 | `fc_only_service_templates`等 | 已完成 |
| Fig. 12 | 滞回寿命-启停Pareto筛查 | `fc_only_service_hysteresis_sweep` | 已完成探索图 |
| Fig. 13 | 冻结慢层在真实留出窗口的oracle验证 | `fc_only_service_holdout_assignment` | 已完成 |
| Fig. 14 | 86,415秒完整留出段跟踪与健康均衡 | `fc_only_full_holdout_replay` | 已完成 |
| Fig. 15 | 留出40 kW新档与N+1容量缺口审计 | `fc_only_holdout_capacity_audit` | 已完成诊断图 |
| Fig. 16 | 30/35/40 kW归一化参考敏感性 | `fc_only_normalization_sensitivity` | 已完成诊断图 |

图文件统一位于`data/results/figures/fc_only_foundation/`，每张图提供320 DPI PNG。

论文收束文件：`docs/PAPER_CLAIM_EVIDENCE_MATRIX.md`、`docs/METHOD_FORMULATION.md`、
`docs/PAPER_INTRO_RELATED_WORK_DRAFT.md`、`docs/PAPER_METHODS_RESULTS_DRAFT.md`；规范数值和
表格位于`data/results/paper_evidence/`；完整留出segment级统计位于
`data/results/fc_only_full_holdout_statistics/`。
