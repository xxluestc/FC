# Codex CLI交接：三堆PEMFC N+1退化感知功率控制

更新时间：2026-07-13

## 1. Git检查点

- 仓库：`H:/其他/2026刘展玮/FC`
- 分支：`codex/gamma-health-foundation`
- 研究代码检查点：`1c90ebd0de3f50439bf00f051d1b8f07d576d0ef`
- 上一论文证据检查点：`f29eeeada607704d30b2246cad4d2d65c2354f8d`
- 完整留出与容量检查点：`d2920cb64a0cc4916169d749cf9fb37193f8d8a5`
- 远端/本地`main`：`7e71f396107751d9d0e93a19da60249e8c8a9122`
- 本文件在研究代码检查点之后单独提交；最终交接提交以`git log -1 --oneline`为准。

研究代码检查点已通过：

- `96`项unittest全部通过；
- baseline lightweight schema check通过；
- `python -m compileall -q src scripts tests`通过；
- `git diff --check`通过；
- Fig.16为`2321 x 1509`、约`320 DPI` PNG；
- `data/results/paper_evidence/`含16张规范CSV表、主张值和源文件SHA-256。

## 2. 当前研究目标与冻结范围

当前主线是三堆PEMFC N+1适度超容量系统：正常正功率需求下两个堆运行、一个堆轮换休息，
研究运行堆选择和在线双堆功率分配。目标是形成基于文献和真实数据、健康随执行动作更新、
可复现且可发表的FC-only内层方法。

当前明确冻结：

- 不使用未来需求功率预测；
- 暂不引入锂电池、SOC或外层氢电分配；
- 不升级TD-MPC、Dreamer、EAWM等高级算法；
- 先关闭物理容量和真实健康观测两个基础门槛。

## 3. 当前框架

### 3.1 数据与负载

主负载来自21UBE0022实车单堆`fc_input_power_kw`。相邻记录间隔超过10秒划分segment，所有
重采样、转移统计和回放均不跨segment。按完整segment最大时间缺口切分：

- 开发：segment 0-21，125,215个1秒样本；
- 留出：segment 22-45，86,415个1秒样本；
- 两者相隔约4.98天；
- Zuo 2024慢变/快变矩阵只作独立压力场景，不与实车1秒矩阵混合。

单堆归一化负载映射到健康双堆聚合参考的95%，保留5%健康容量余量。冻结主分析使用30 kW
开发参考；30 kW只是开发分区目标最大值，不是已确认额定功率。

### 3.2 动作驱动健康闭环

每个电堆状态包括累计退化$D_j$、上一步电流、开关状态和驻留时间。执行动作后依次完成：

`实际离散电流/开关 -> 连续与事件退化 -> D -> theta -> IV/功率 -> 下一步候选能力`

退化含连续运行、变载、运行启动和预声明异质性；健康映射到LZW参数
`theta=[i0, ih, R_ohm]`，再进入半经验IV模型。该状态是跨数据链动作驱动预测态，尚不是
21UBE0022同车同堆观测posterior。

### 3.3 功率调控

慢层主方法是24小时`health-greedy`：选择当前累计退化最小的两个堆，使最老堆休息，再把开发
模板中暴露更重的角色分配给剩余健康裕度更大的堆。它只读取当前健康和开发暴露均值。

秒级快层为`Instant-health`：在慢层指定在线双堆中枚举离散电流
`{0,25,90,120,160,195,270,370} A`，最小化氢耗代理、单步退化、性能损失、跟踪误差、启停和
变载代价。硬约束为两个堆在线、5.5 kW跟踪容差和最小驻留；严格驻留无可行动作时允许有记录
的安全覆盖。精确FC跟踪剪枝与穷举在测试状态/需求上返回相同动作、目标和约束结果。

Expected-max和Gamma-CVaR保留为慢层消融；Gamma单秒样本不进入快层策略排序，随机性放在小时级
聚合暴露和风险评价。

## 4. 已完成结果

### 4.1 长期开发结果

实车标定模板下平均健康边界到达时间：

| 策略 | 小时 |
|---|---:|
| fixed pair | 1287.45 |
| health-greedy | 1680.80 |
| Expected-max | 1682.40 |
| Gamma-CVaR | 1646.10 |

health-greedy相对固定双堆增加393.35 h，即30.553%。在经验Markov、Zuo慢变、Zuo快变中分别
增加376.8、401.6、231.45 h。11组单因素边界设置全部10/10种子正增益，最小单种子增益259 h。
这些数字是到LZW标定健康边界的时间，不是失效寿命或RUL。

Expected-max相对health-greedy仅增加1.5-3.7 h且获胜率20%-30%；Gamma-CVaR无稳定优势。
四种冻结开发暴露乘三种健康身份的12个决策点枚举表明，三策略12/12选择相同在线集合，且
Expected-max和Gamma-CVaR两类目标的最大遗憾均为0；health-greedy仅4/12角色顺序不同。

### 4.2 冻结窗口与完整留出

- 18个冻结中心窗口案例全部可执行；
- health-greedy的oracle在线集合命中率100%，最大退化遗憾0；
- 完整回放覆盖24个segment、86,415秒、3种健康身份、2种策略；
- 共144例、518,490个闭环步，144/144完成；
- 硬约束违规0，最大跟踪误差5.498756 kW；
- 455个有审计驻留安全覆盖步，占0.088%。

当固定集合包含最老堆时，health-greedy在两个非平凡健康身份中均8/8运行segment降低终端最大
退化，均值差为-0.009619156和-0.009930907个百分点。以8个完整segment为统计单位：

- 95% bootstrap区间为`[-0.014610982,-0.005262698]`和
  `[-0.015212701,-0.005394733]`；
- 单侧精确Wilcoxon原始`p=0.00390625`，Holm校正后均`p=0.0078125`；
- 逐段删一后全部保留显著，最大校正`p=0.015625`；
- 16个单段符号反转压力情景有6个失去显著性，说明外推仍受`n=8`限制。

该改善不是Pareto占优：冻结30 kW下总期望退化分别增加0.005508464和0.006374698个百分点，
跟踪MAE约增加0.035 kW，氢耗强度则下降。

### 4.3 归一化、容量和容差边界

30/35/40 kW全量事后敏感性使用相同留出、健康身份、权重和5.5 kW容差：

| 参考 | 正功率截峰 | 完成/总数 | 硬违规 | 最老堆0均值差 | 最老堆1均值差 |
|---:|---:|---:|---:|---:|---:|
| 30 kW | 11.585% | 144/144 | 0 | -0.009619 | -0.009931 |
| 35 kW | 2.672% | 144/144 | 0 | -0.009864 | -0.010621 |
| 40 kW | 0% | 144/144 | 0 | -0.009936 | -0.011331 |

六个主比较均8/8改善、95%区间低于0、Holm校正显著。因此最差堆改善方向不依赖30 kW截峰，
但总期望退化差在参考/身份间变号，仍不构成Pareto占优。

独立容量审计表明，若30 kW映射不截峰，最坏健康/策略组合有6.648%正功率步超过双堆初始
物理容量；允许5.5 kW包络后仍有2.759%。事后严格参考下界为36.756035 kW。40 kW只是覆盖
当前峰值的候选，不能替代车辆额定资料。

跟踪容差定向审计使用冻结最大误差案例segment 42：4.90 kW及以下失败，4.95 kW成功且实际
最大误差4.941 kW。5.5 kW相对当前最小测试成功值有0.55 kW余量；这不是全留出容差扫描。

## 5. 论文与结果资产

优先阅读：

1. `docs/MASTER_RESEARCH_PLAN.md`
2. `docs/PROJECT_STATUS_BOARD.md`
3. `docs/METHOD_STRATEGY_DECISION_2026-07-13.md`
4. `docs/METHOD_FORMULATION.md`
5. `docs/PAPER_CLAIM_EVIDENCE_MATRIX.md`
6. `docs/PAPER_INTERNAL_REVIEW.md`
7. `docs/PAPER_INTRO_RELATED_WORK_DRAFT.md`
8. `docs/PAPER_METHODS_RESULTS_DRAFT.md`
9. `docs/PAPER_ABSTRACT_CONCLUSION_DRAFT.md`

规范证据：

- `data/results/paper_evidence/claim_values.json`：正文规范数值；
- `data/results/paper_evidence/source_manifest.json`：输入SHA-256；
- `data/results/paper_evidence/table01_*.csv`至`table16_*.csv`；
- `data/results/fc_only_full_holdout_statistics/`：bootstrap、Wilcoxon、影响力；
- `data/results/fc_only_normalization_sensitivity/`：30/35/40 kW诊断；
- `data/results/fc_only_tracking_tolerance_audit/`：最坏案例容差边界；
- `data/results/fc_only_service_objective_audit/`：风险目标遗憾。

论文图统一位于`data/results/figures/fc_only_foundation/`。Fig.1-16均为PNG；Fig.16是
30/35/40 kW归一化敏感性，图注见`figure_manifest.md`。不生成PDF图。

核心本地文献位于`G:/大论文/AI文献库`，本轮直接使用：

- `多模块功率调控/zuo2024.txt`
- `多模块功率调控/ghaderi2022.txt`
- `多模块功率调控/功率分配/qlearning2023.txt`
- `多模块功率调控/li2025iosl.txt`
- `多模块功率调控/功率分配/tumer2025.txt`

BibTeX元数据见`docs/PAPER_CORE_REFERENCES.bib`。

## 6. 复现顺序

PowerShell：

```powershell
cd H:\其他\2026刘展玮\FC
$env:PYTHONPATH='src'

python scripts/00_materialize_key_data.py
python -m unittest discover -s tests -v
python scripts/run_baseline.py --mode check
python -m compileall -q src scripts tests
git diff --check
```

论文证据审计：

```powershell
$env:PYTHONPATH='src'
python scripts/47_bootstrap_full_holdout_effects.py
python scripts/48_audit_segment_influence.py
python scripts/49_summarize_normalization_sensitivity.py
python scripts/50_audit_tracking_tolerance_boundary.py --resume
python scripts/51_audit_service_objective_regret.py
python scripts/46_build_paper_evidence_tables.py
```

若必须从头重跑归一化敏感性，先执行：

```powershell
$env:PYTHONPATH='src'
python scripts/44_replay_full_real_holdout_segments.py --jobs 8
python scripts/44_replay_full_real_holdout_segments.py --jobs 9 `
  --normalization-power-kw 35 --out-dir data/results/fc_only_full_holdout_norm35 --skip-plot
python scripts/44_replay_full_real_holdout_segments.py --jobs 9 `
  --normalization-power-kw 40 --out-dir data/results/fc_only_full_holdout_norm40 --skip-plot
```

35/40 kW全量回放计算较重，结果已提交；除非上游物理口径或控制器改变，不要重复运行。
`scripts/50... --resume`会复用已写容差档；上游变化时应删除`--resume`并全量重跑该定向审计。

## 7. 不可声称内容

- 不称三堆两运行N+1结构或健康感知EMS为首次提出；Zuo 2024等已有相关框架。
- 不称LZW健康边界为失效阈值，不称边界到达时间为真实寿命/RUL。
- 不称当前健康为21UBE0022实车posterior或已完成在线观测校正。
- 不称30 kW或40 kW为已验证物理额定功率。
- 不称完整留出验证了动态24小时重复调度；最长segment仅7.543 h。
- 不称映射的单堆实车轨迹为真实三堆硬件/HIL验证。
- 不称health-greedy在全部成本上Pareto占优。
- 不称方法使用未来需求预测、锂电池或SOC优化。
- 不把Gamma短时稀疏采样诊断写成21UBE0022本车独立辨识的普适定律。

## 8. 当前阻塞输入

需要用户或车辆资料确认：

1. 21UBE0022燃料电池额定净功率，或控制器允许的FC目标/输出上限是否为40 kW；最好提供铭牌、
   控制器标定表、车辆技术参数或原始字段说明。
2. 可与21UBE0022时间轴和电堆身份绑定的MAT健康变量链：变量名、时间戳、车辆/堆编号、采样率、
   单位及噪声/滤波来源。当前`x_est`来自另一条6104点链，不能直接逐行对齐。

本地全文检索未找到上述两个物理证明。获得资料后：

- 按确认额定参考重建负载映射，执行无截峰最终回放和容量审计；
- 在现有动作驱动预测态上加入观测校正接口，报告预测-观测偏差和不确定性；
- 再决定是否进入锂电池外层和未来需求预测。

## 9. 下一执行顺序

1. 保持`1c90ebd...`研究代码检查点不变，先取得额定功率和MAT身份链。
2. 若资料暂时无法取得，继续做期刊选择、正式中文/英文成稿、完整引用核查和图表排版，但所有
   物理边界保持显式。
3. 取得额定功率后重跑物理归一化主链，不用留出峰值反向调参。
4. 取得MAT链后实现确定性预测+观测校正，再做Gamma观测不确定性。
5. 两个基础门槛关闭后，才启动锂电池外层；未来需求预测作为更后面的独立优化点。

## 10. 代理协作

本地`claude`命令实际接DeepSeek，可作为普通只读/窄任务代理，不设预算。例如：

```powershell
claude -p --effort max --permission-mode plan --tools "Read,Glob,Grep" `
  --no-session-persistence "只读审查指定文件并给出可验证问题，不修改仓库。"
```

代理适合文献字段提取、结果一致性检查和初稿挑错；物理口径、统计方法、策略升级和最终代码
必须由主代理复核。本轮两次代理调用中，一次给出有效审稿意见，一次只返回空泛完成信息；后者
未被采纳，也没有代理直接修改文件。
