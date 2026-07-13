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
| `fig15_holdout_capacity_shift_audit` | 开发/留出单堆负载持续曲线、候选归一化参考的截峰比例，以及最坏健康/策略组合下不截峰时超过双堆容量的留出比例。36.76 kW为使用留出峰值计算的事后诊断下界；40 kW仅为待物理资料确认的控制目标候选。 |

每张图保存为320 DPI PNG。图内`30 s*`表示工程时间基准假设，不是Zuo论文直接给定值。
