# FC-only基础结果图清单

| 文件 | 建议图注 |
|---|---|
| `fig01_markov_timescale_audit` | 实车经验Markov矩阵在不同采样间隔下的等效状态变化率，以及暂按30秒解释的Zuo慢变/快变压力场景。下采样会漏掉中间变化，因此不同时间基准的矩阵不直接融合。 |
| `fig02_deterministic_tradeoff_forest` | Instant-health相对Average在10个配对负载种子上的功率跟踪、单位输出电量氢耗和期望退化变化。点为配对均值，误差线为95%区间；负值表示降低。 |
| `fig03_real_holdout_validation` | 冻结参数在segment 22-45固定中心窗口上的执行成功率、正功率成功窗口跟踪误差和规划时间。Average在两个窗口无严格等电流可行动作。柱为均值，误差线为跨正功率窗口标准差。 |
| `fig04_gamma_timescale_diagnostic` | 当前Gamma标定在不同聚合暴露时长下的有效shape，以及采样增量低于条件均值1%的概率。120秒尺度几乎必然近零，说明短时在线样本不适合估计Gamma均值。 |
| `fig05_aggregate_gamma_sensitivity` | 历史诊断图。冻结120秒动作暴露重复至1000小时会同步放大启停事件，不能作为寿命预测或方法效果证据。仅用于说明Gamma聚合时间尺度。 |
| `fig06_mechanism_ablation` | 冻结策略动作路径下的75点机制消融。上行为Instant相对Average，下行为Instant相对Rotating；负值表示总退化降低。独立色标用于分别显示小幅脆弱差异和事件主导差异。 |
| `fig07_service_horizon_screen` | 小时级N+1调度开发筛查。左图为20个配对开发种子的健康边界生存曲线，右图为中位数及P10-P90。健康边界是LZW标定轨迹终点，不是失效阈值。 |
| `fig08_reschedule_sensitivity` | Expected-max与Gamma-CVaR在6/12/24/48小时最小重调度周期下的健康边界时间和平均启动次数。延长周期显著减少启动而没有消除相对固定双堆的收益。 |
| `fig09_service_gamma_cv_sensitivity` | Gamma终点CV 5%/10%/20%开发敏感性。左图为策略健康边界时间均值与标准差，右图为相对固定双堆的配对增益均值及最小-最大范围。 |
| `fig10_service_robustness` | 扩展实车模板上的单因素稳健性。分别改变健康边界、连续退化、变载、完整校准段运行启停率和调度启动系数；线为10种子配对平均增益，阴影为最小-最大范围。 |
| `fig11_expanded_service_results` | 120个快层闭环模板的暴露构成，以及四类开发场景20种子到健康边界时间和相对固定双堆的配对范围。实车主模板包含完整校准段启停率；三类压力场景保持连续正功率。 |
| `fig12_service_hysteresis_pareto` | 切换滞回的开发Pareto筛查。较大阈值可减少主动启动，但没有阈值通过冻结的逐种子非劣规则，因此不进入主方法。 |
| `fig13_holdout_assignment_validation` | 冻结慢层在未参与标定的真实中心窗口上的事后oracle验证。三个健康身份循环共18例；health-greedy与Expected-max均命中oracle在线集合且零最大退化遗憾，Expected-max的角色顺序命中未转化为额外健康收益。虚线为5.5 kW冻结跟踪容差。 |
| `fig14_full_real_holdout_replay` | 冻结health-greedy在segment 22-45全部完整连续块上的回放。左图为最长留出段的60秒聚合需求与FC输出，中图为8个正功率段最大跟踪误差，右图为不同最老堆身份下相对固定双堆的终端最大退化变化。段内状态连续，段间不跨未知缺口。 |
| `fig15_holdout_capacity_shift_audit` | 开发/留出单堆负载持续曲线、候选归一化参考的截峰比例，以及最坏健康/策略组合下不截峰时超过双堆容量的留出比例。36.76 kW为使用留出峰值计算的事后诊断下界；后续全年独立审计支持40 kW作为经验运行参考，但不确认铭牌额定值。 |
| `fig16_normalization_sensitivity` | 冻结30 kW主分析与35/40 kW事后全量诊断的截峰比例、最大跟踪误差、终端最大退化配对差及95% segment-bootstrap区间和总期望退化权衡。35/40 kW不用于确认物理额定功率。 |
| `fig17_synthetic_health_observer` | 健康观测接口的合成诊断。预声明15%异质性条件均值作为合成真值，每24小时输入一次直接退化代理；校正belief减小注入的模型漂移。该图只验证predict/execute/correct时序和不确定性记录，不是21UBE0022实车posterior证据。 |
| `fig18_21ube0022_power_envelope` | `21UBE0022_苏E02625F`全年400个CSV的运行功率包络和分月电堆侧上尾。目标功率p99/p99.9均为40 kW；最大值只作经验观测，不解释为铭牌额定净功率。 |
| `fig19_21ube0022_voltage_health_signal` | 固定195/270/340 A和65--70 C后的总堆电压、跨电流档时期台阶，以及最终时期环境校正前后Theil--Sen趋势。环境校正后95%区间包含0，因此该图是健康信号可辨识性审计，不是SOH曲线。 |
| `fig20_external_monthly_replay` | 冻结40 kW方法在2025-06至2026-06独立实车功率块上的跨月回放。每月三层、每层30分钟；左图为功率覆盖，中图为health-greedy相对固定双堆的月级最大退化变化，右图显示最大退化改善与总退化小幅增加的取舍。健康和控制状态逐块重置，不解释为同一电堆连续13个月老化。 |
| `fig21_n_plus_one_service_boundary` | 200个配对开发种子下的首/第二标定边界。虚线为首个堆边界、实线为N+1第二堆边界；Blend 0.50兼顾投影最大与第二大损伤，纯N+1目标展示极早牺牲一个堆的代价。边界是LZW标定终点，不是物理失效。 |
| `fig22_n_plus_one_cross_month` | 冻结Blend 0.50在2025-06至2026-06外部月度暴露上的长期验证。左、中图分别为相对固定双堆的月均首边界和N+1第二边界增益；右图为13个月BCa均值区间，并以纯N+1作极端对照。权重未用外部月份重调。 |
| `fig23_n_plus_one_parameter_robustness` | 41个预声明参数场景、每场景20个配对种子的N+1鲁棒性。左图比较首/第二边界取舍，中图按参数类别汇总原始Blend，右图给出其最差第二边界场景及95%区间。GP随机效应为文献压力分位数，不是本车拟合。 |
| `fig24_two_stage_continuity_screen` | 两阶段RUL和第二边界连续性目标的定向否证。解析投影会增加切换，并在异质性错配下牺牲N+1第二边界；该图保留为负方法证据，不作为主结果。 |
| `fig25_protected_blend_screen` | 局部baseline-protection筛查。候选动作同时受一步最大损伤和第二大损伤约束，但长期轨迹仍在两个异质性场景出现第二边界负95%区间，因此不能作为安全保证。 |
| `fig26_guarded_blend_cross_month` | Guarded Blend在13个月外部真实暴露下的异质性验证。参考场景保留Blend收益；强随机效应对齐场景第二边界区间跨0；强错配场景精确回退固定双堆。健康逐块重置，因子不是车辆拟合值。 |
| `fig27_voltage_loss_boundary_mapping` | LZW损伤代理到定工况电压损失阈值的映射。LZW标定终点不等于物理EOL，5%及以上阈值只作外推压力边界。 |
| `fig28_frozen_process_physical_boundaries` | 固定退化过程后改变停止边界的压力测试，用于区分边界选择和退化过程缩放。 |
| `fig29_blend_weight_strong_heterogeneity` | 强同向异质性下五个Blend权重的否证筛查。单纯调权没有通过N+1第二边界非负门槛。 |
| `fig30_rate_bounded_blend_audit` | Rate-bounded Blend在参数、物理边界和跨月模板上的三层审计。1.10门槛来自既有名义异质性包络。 |
| `fig31_lzw_overall_exposure_screen` | LZW总体暴露退化模型停止门筛查。负载项相对运行时钟没有稳定增益，因此只保留终点与趋势，不进入动作分辨退化率。 |
| `fig32_simplified_random_load_comparison` | 三类随机动态负载下五种功率分配策略的闭环比较，以及Instant考虑健康前后的同种子消融。当前只证明简化退化链路跑通，不作寿命结论。 |

每张图保存为320 DPI PNG。图内`30 s*`表示工程时间基准假设，不是Zuo论文直接给定值。
