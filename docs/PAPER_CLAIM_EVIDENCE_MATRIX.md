# 论文主张与证据矩阵

更新时间：2026-07-13

## 1. 论文定位

研究问题：在不预测未来需求、暂不引入锂电池外层的条件下，如何把实车单堆负载、动作驱动
健康演化和三堆N+1选择/分配组成可审计的双时间尺度闭环，并验证健康均衡是否能在强基线和
未见真实连续块上成立。

当前最稳妥的贡献表述不是“提出首个三堆健康EMS”，而是：

1. 构建实车负载时间隔离的三堆N+1测试链，并显式审计归一化和容量外推；
2. 建立每步动作驱动的 $D\rightarrow\theta\rightarrow IV/功率$闭环，避免健康只进入静态代价；
3. 提出无未来预测的24小时health-greedy慢层与秒级Instant快层耦合，并区分运行启停与调度启停；
4. 证明单秒Gamma采样不适合短时策略排序，把随机性移到小时级聚合风险层；
5. 通过强基线、机制敏感性、冻结oracle窗口和全部未见完整segment验证，报告有效区间与失败边界。

## 2. 与本地核心文献的边界

| 文献 | 已有贡献 | 本项目不能声称 | 本项目差异 |
|---|---|---|---|
| Zuo et al., RESS 2024, DOI `10.1016/j.ress.2023.109660` | 三堆超容量、两堆同时运行、随机退化、事件负载分配和启停选择 | N+1三堆结构或随机退化EMS首创 | 实车时间隔离负载、动作到IV闭环、Gamma时间尺度诊断、强基线和冻结真实留出 |
| Ghaderi et al., IEEE TVT 2022, DOI `10.1109/TVT.2022.3167319` | 三堆+电池、博弈EMS、在线系统辨识、健康感知 | 在线健康感知或多堆EMS首创 | 当前先解FC-only内层，强调每步机理健康闭环和未见连续块验证 |
| Ghaderi et al., ECM 2023, DOI `10.1016/j.enconman.2023.117524` | 三层Q-learning多堆混动EMS、在线估计、双循环对照 | RL健康EMS首创 | 不用预测/RL，把复杂度放在可审计健康和时间尺度边界 |
| Li et al., IEEE TTE 2025, DOI `10.1109/TTE.2024.3491107` | 多堆机车系统寿命与氢耗/电池等效成本集成、HIL | 系统寿命集成优化首创 | 当前健康边界更保守，不冒充失效寿命，且以实车单堆负载映射验证 |
| Tümer et al., C&CE 2025, DOI `10.1016/j.compchemeng.2025.109142` | 双堆实时最优分配、RLS-KF电压参数在线估计、退化自适应 | 电压在线健康更新首创 | 当前实车observer尚未完成；贡献只能是动作预测态闭环和N+1双时间尺度验证 |

本地文本：`G:/大论文/AI文献库/多模块功率调控/zuo2024.txt`、`ghaderi2022.txt`、
`li2025iosl.txt`、`功率分配/qlearning2023.txt`、`功率分配/tumer2025.txt`。

## 3. 主张-证据-边界

| ID | 可写主张 | 定量证据 | 生成脚本/结果 | 论文边界 | 状态 |
|---|---|---|---|---|---|
| C1 | 健康随每步实际动作单调更新并反馈IV/功率 | 108项测试含在线健康携带、老化功率下降、异质性、观测接口隔离和外部块不跨缺口 | `gamma_process.py`、`mechanistic.py`、`external_validation.py`、`tests/` | 主实验为预测态代理，不是真实posterior | 可用 |
| C2 | 实车负载标定与留出按完整segment时间隔离 | 标定125,215行，留出86,415行，间隔约4.98天 | `load_zuo_calibration/`、Fig.1 | 30 kW不是额定值 | 可用 |
| C3 | 单秒Gamma样本不适合短时策略均值排序 | 120秒有效shape约$10^{-3}$，近零概率接近1 | `fc_only_gamma_timescale/`、Fig.4 | 不是否定Gamma长期建模 | 可用 |
| C4 | 简单health-greedy是当前最强且稳健的慢层主方法 | 实车开发模板均值1680.80 h；固定1287.45 h；11组单因素均10/10正增益 | `fc_only_service_robustness/`、Fig.10-11 | 时间到LZW边界，不是真实寿命 | 可用 |
| C5 | Expected-max/Gamma-CVaR没有稳定超过强health-greedy | Expected-max仅+1.5至+3.7 h，获胜率20%-30%；Gamma-CVaR更低；12个预声明决策点三者在线集合一致且两类目标遗憾均为0 | `fc_only_service_scheduler_strong_baseline_*`、`fc_only_service_objective_audit/`、Fig.11 | 自身目标最优只是实现一致性，不包装成性能创新 | 可用 |
| C6 | 冻结慢层选择在未见窗口命中最优在线集合 | 18/18可行，health-greedy oracle集合命中100%，最大遗憾0 | `fc_only_service_holdout_assignment/`、Fig.13 | 只验证选择，不代替全段回放 | 可用 |
| C7 | 冻结双时间尺度方法可执行于全部未见完整段 | 144/144例、518,490步、零违规、455个有审计驻留覆盖步、最大误差5.499 kW | `fc_only_full_holdout_replay/`、Fig.14 | 最长段7.543 h；验证入口选择和段内快层，不是动态24 h重调度；30 kW以上被截峰 | 可用 |
| C8 | health-greedy降低最差堆健康代价而非Pareto支配 | 原留出最老堆为0/1时均8/8段改善；独立跨月回放13/13个月改善，月级BCa 95%区间[-0.004063830, -0.003848424]个百分点；总退化小幅增加 | `fc_only_full_holdout_statistics/`、`fc_only_external_monthly_replay/` | 跨月块仍来自同一车辆且逐块重置健康；同时报告总退化代价 | 可用 |
| C9 | 最差堆改善方向在30/35/40 kW参考下保持；全年数据支持40 kW经验运行参考 | 截峰11.585%/2.672%/0%；目标40 kW出现291,209次，全年目标p99/p99.9均为40 kW；六个主比较均8/8改善 | `fc_only_normalization_sensitivity/`、`liu_21ube0022_identity_rating/`、Fig.15-16、18 | 40 kW不是铭牌额定净功率；30 kW保留冻结敏感性 | 可用，需限称谓 |
| C10 | 健康观测软件接口严格，但实车信号尚不能辨识为累计退化 | 8项observer测试；合成RMSE降低81.288%；263.6万匹配电压行识别两个时期台阶；环境校正趋势95% CI含0 | `synthetic_health_observer/`、`liu_21ube0022_voltage_health_audit/`、Fig.17、19 | 旧MAT只作先验；实车电压只作性能残差，不得声称SOH posterior | 软件可用、实车回写否决 |
| C11 | 5.5 kW是有动态可行性依据的工程容差而非物理常数 | 冻结最大误差案例中4.90 kW失败、4.95 kW成功；5.5 kW余量0.55 kW | `fc_only_tracking_tolerance_audit/` | 仅最坏案例定向审计，不是全留出容差扫描 | 可用作边界解释 |

## 4. 结果表规划

| 表 | 内容 | 直接数据源 | 是否需要新实验 |
|---|---|---|---|
| Table 1 | 三堆、动作网格、退化系数、健康边界、负载参考、调度周期 | `lzw_gamma_calibration.json`和配置类 | 否 |
| Table 2 | 固定、health-greedy、Expected-max、Gamma-CVaR四场景长期强基线 | `fc_only_service_scheduler_strong_baseline_*/summary.csv` | 否 |
| Table 3 | 11组机制/边界稳健性最小-均值-最大配对增益 | `fc_only_service_robustness/` | 否 |
| Table 4 | 冻结窗口oracle命中、遗憾和跟踪 | `fc_only_service_holdout_assignment/summary.csv` | 否 |
| Table 5 | 全部留出段可行性、健康均衡、氢耗和跟踪权衡 | `fc_only_full_holdout_replay/` | 否 |
| Table 6 | 30/31.343/40 kW参考的截峰和容量超限 | `fc_only_holdout_capacity_audit/normalization_capacity_audit.csv` | 否 |
| Table 7 | 三种健康身份的完整segment bootstrap点估计和95%区间 | `fc_only_full_holdout_statistics/segment_bootstrap_summary.csv` | 否 |
| Table 8 | 两个预声明最差堆主检验及Holm校正 | `fc_only_full_holdout_statistics/primary_wilcoxon_tests.csv` | 否 |
| Table 9-10 | 逐段删一和单段符号反转影响力审计 | `fc_only_full_holdout_statistics/` | 否 |
| Table 11-13 | 30/35/40 kW参考的可行性、代价和segment统计 | `fc_only_normalization_sensitivity/` | 已完成事后诊断 |
| Table 14 | 冻结最大误差案例的4.0-5.5 kW定向容差扫描 | `fc_only_tracking_tolerance_audit/` | 已完成定向诊断 |
| Table 15-16 | 12个慢层决策点的自身目标遗憾汇总和分配明细 | `fc_only_service_objective_audit/` | 已完成决策诊断 |

## 5. 建议论文结构

1. Introduction：多堆冗余价值、健康闭环缺口、真实负载与时间尺度问题；列出五项贡献。
2. Related Work：多堆分配、健康感知EMS、随机退化、在线observer；明确不争夺已有首创。
3. Problem Formulation：三堆两运行、FC-only边界、负载映射、状态/动作和评价边界。
4. Method：动作健康转移、$D\rightarrow\theta\rightarrow IV$、Instant快层、health-greedy慢层、Gamma聚合层。
5. Experimental Protocol：时间切分、120开发模板、强基线、配对统计、冻结留出和容量审计。
6. Results：先开发稳健性，再oracle选择，最后完整真实段与代价权衡。
7. Discussion：为什么复杂风险目标无稳定增益、Gamma时间尺度、30 kW容量限制、observer缺口。
8. Conclusion：只总结已证实范围，不提前引入电池和需求预测。

## 6. 当前发表门槛

| 优先级 | 门槛 | 当前状态 | 处理 |
|---:|---|---|---|
| 1 | 功率归一化/容量依据 | 全年实车支持40 kW经验参考；铭牌额定净功率仍未知 | 最终链用40 kW无截峰；30 kW保留敏感性；不写铭牌额定值 |
| 2 | 实车健康observer | predict/correct与合成验收完成；旧MAT和长期电压均不能直接观测`D` | 保持动作驱动预测态；电压只作时期化性能残差，等待堆编号/维护记录或可验证测量模型 |
| 3 | 主方法新颖性表述 | 算法本身简单，集成与审计证据较强 | 以机制闭环和验证协议为贡献，不虚构复杂算法 |
| 4 | 论文表格与数值追踪 | 16张规范表、主张值和源文件哈希已统一导出 | 上游结果变化时重跑脚本46 |
| 5 | 正式稿 | Intro/Related Work/Methods/Experiments/Results/Discussion已有中文初稿 | 已按首轮内部审稿修订容量与统计边界 |

## 7. 禁止写入摘要的表述

- “首次提出三堆两运行架构”或“首次健康感知多堆EMS”；
- “显著延长真实燃料电池寿命”或“准确预测RUL”；
- “真实健康在线观测校正已完成”；
- “在完整原始峰值轨迹上零违规”，除非物理额定参考确认并重跑；
- “同时优化燃料电池和锂电池”或“利用未来需求预测”；
- “所提方法在全部成本指标上优于基线”。
