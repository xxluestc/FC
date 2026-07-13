# FC-only基础结果图清单

| 文件 | 建议图注 |
|---|---|
| `fig01_markov_timescale_audit` | 实车经验Markov矩阵在不同采样间隔下的等效状态变化率，以及暂按30秒解释的Zuo慢变/快变压力场景。下采样会漏掉中间变化，因此不同时间基准的矩阵不直接融合。 |
| `fig02_deterministic_tradeoff_forest` | Instant-health相对Average在10个配对负载种子上的功率跟踪、单位输出电量氢耗和期望退化变化。点为配对均值，误差线为95%区间；负值表示降低。 |
| `fig03_real_holdout_validation` | 冻结参数在segment 22-45固定中心窗口上的执行成功率、正功率成功窗口跟踪误差和规划时间。Average在两个窗口无严格等电流可行动作。柱为均值，误差线为跨正功率窗口标准差。 |
| `fig04_gamma_timescale_diagnostic` | 当前Gamma标定在不同聚合暴露时长下的有效shape，以及采样增量低于条件均值1%的概率。120秒尺度几乎必然近零，说明短时在线样本不适合估计Gamma均值。 |
| `fig05_aggregate_gamma_sensitivity` | 历史诊断图。冻结120秒动作暴露重复至1000小时会同步放大启停事件，不能作为寿命预测或方法效果证据。仅用于说明Gamma聚合时间尺度。 |
| `fig06_mechanism_ablation` | 冻结策略动作路径下的75点机制消融。上行为Instant相对Average，下行为Instant相对Rotating；负值表示总退化降低。独立色标用于分别显示小幅脆弱差异和事件主导差异。 |

每张图保存为320 DPI PNG。图内`30 s*`表示工程时间基准假设，不是Zuo论文直接给定值。
