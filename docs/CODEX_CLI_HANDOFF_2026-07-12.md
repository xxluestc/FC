# FC 项目 Codex CLI 交接文档

更新时间：2026-07-12
目标仓库：`https://github.com/xxluestc/FC.git`
远端基线：`main = 7e71f396107751d9d0e93a19da60249e8c8a9122`
工作分支：`codex/gamma-health-foundation`
已验证代码检查点：`1b8e8de`（Audit degradation coefficient and objective scaling）

## 1. 给接手 Codex 的第一条指令

不要从头设计，不要重做已经完成的实验，也不要执行 `git reset --hard`、`git checkout --` 或覆盖当前工作树。先执行：

```bash
git branch --show-current
git log --oneline --decorate -10
git status --short --untracked-files=all
git show --stat --oneline 1b8e8de
```

确认代码至少包含 `1b8e8de`。如果只拿到远端 `main`，应按编号依次应用交接补丁：

```bash
git switch -c codex/gamma-health-foundation
git am /path/to/full_series/*.patch
```

然后运行：

```bash
python scripts/00_materialize_key_data.py
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python scripts/run_baseline.py --mode check
python -m compileall -q src scripts tests
git diff --check
```

交接时最新验证结果：61项测试通过；baseline轻量检查通过；compileall和`git diff --check`通过。

## 2. 必须保护的工作树边界

`data/key/li_junhao_dual_stack/engine5_three_day_processed.csv.gz`在网页版环境中存在一个来源不明的本地二进制改动。该文件没有进入任何提交，也不应进入补丁。

接手时：

- 不要暂存、还原或删除这个文件；
- 先比较本地文件与远端版本来源，再决定是否重新物化；
- 不要为了得到“干净状态”而清空整个工作树；
- 大型`testbed_trajectory.csv`是可再生实验轨迹，默认被`.gitignore`排除。

## 3. 项目最终研究目标

构建多堆PEMFC/动力电池退化感知预测功率调控框架：

```text
当前需求和短时未来需求
+ 电池SOC/功率状态
+ 每个电堆当前D/theta/SOH、功率、电流、开关和驻留状态
        ↓
候选多堆动作 [I1...In, on/off1...n]
        ↓
机理世界模型逐步递推
P_dem = sum(P_fc_i) + P_bat
D_i(k+1) = D_i(k) + DeltaD_i(action,event)
theta_i(k+1) = g_i(D_i(k+1))
SOC(k+1) = f(SOC(k), P_bat)
        ↓
氢耗、退化、电池负担、性能损失、SOC、切换和约束
        ↓
规划/学习策略选择第一步动作
        ↓
执行、接收新观测、修正健康状态、进入下一步
```

当前已经完成“无新观测时的在线健康预测闭环”。真实传感观测下的UKF/PF posterior修正尚未完成，不能把预测态称为观测后健康估计。

## 4. 已提交且完成的工作

### 4.1 提交历史

| 提交 | 内容 | 状态 |
|---|---|---|
| `7feb6dd` | 项目追踪文档、Gamma健康状态基础 | 完成 |
| `22eb50c` | LZW 6104事件Gamma标定、theta映射、动态性能损失代理 | 完成 |
| `e5fcd8a` | 多堆机理世界模型、Instant/Beam、安全投影 | 完成 |
| `6f64179` | 真实需求段健康感知多堆公平基准 | 完成 |
| `33cf379` | 事件条件化概率功率预测审计 | 候选保留，不替换默认预测器 |
| `bd117d7` | 随机/真实块负载、在线健康不变量、双/三堆统一测试框架 | 完成 |
| `1b8e8de` | 退化系数敏感性、目标尺度审计、单步退化归一化实验 | 完成代码与烟雾验证；评价协议仍需修正 |

### 4.2 在线健康状态闭环

核心文件：

- `src/fc_power/health/gamma_process.py`
- `src/fc_power/health/lzw_gamma_calibration.py`
- `src/fc_power/health/dynamic_proxy.py`
- `src/fc_power/world_model/mechanistic.py`
- `src/fc_power/world_model/lzw_factory.py`

每个决策步执行：

```text
state.D/theta/current/on/dwell
→ action current/on
→ Gamma条件均值或随机正增量
→ 新D
→ theta(D)
→ IV模型计算V/P
→ 氢耗、电池补偿、SOC和约束
→ next_state
```

测试框架强制检查：

- `DeltaD >= 0`；
- `D_after = D_before + DeltaD`；
- 下一步`D_before`等于上一步`D_after`；
- 功率平衡误差在数值容差内；
- SOC、电池功率、电堆动作、驻留规则可审计；
- 固定随机种子下Gamma轨迹可复现；
- 双堆和三堆均可执行。

### 4.3 Ghaderi/Pei系数口径

论文公式和代码最终统一为：

- 0 A带电怠速：只计`alpha_low`；
- 正电流运行：计`alpha_on`自然衰减；
- 370 A高载：`alpha_on + alpha_high`；
- 启停与变载：作为可叠加离散事件；
- 完全停机：连续退化率为0；
- 堆间异质性通过`heterogeneity_factor`缩放。

默认文献值：

```text
start/stop  0.00196 %/cycle
high load  0.001470 %/hour
low load   0.00126 %/hour
natural on 0.002 %/hour
load shift 5.93e-5 %/cycle
```

这些系数来自Ghaderi 2023 Table 4（引用Pei 2008），没有被LZW数据独立因果辨识。LZW 6104事件theta轨迹只用于累计损伤尺度到`[i0, ih, R_ohm]`的单调映射。

### 4.4 随机动态负载和统一测试框架

核心文件：

- `src/fc_power/evaluation/load_profiles.py`
- `src/fc_power/evaluation/multistack_testbed.py`
- `scripts/23_run_multistack_testbed.py`
- `tests/test_load_profiles.py`
- `tests/test_multistack_testbed.py`

包含两类负载：

1. 事件标注半马尔可夫压力负载：idle/cruise/high/braking；
2. 实测功率连续块重采样：保留块内真实动态，降低拼接边界跳变。

合成负载是透明压力测试假设，不代表某辆车的真实工况分布。实测块重采样打破跨块行程语义，也不能替代完整真实行程。

测试框架支持：Average、Rotating、Instant-health、Beam-health；双堆和三堆；确定性均值健康与随机Gamma健康；共同SOC恢复尾段；同种子配对差值、相对变化、95% CI和改善率。

曾发现种子2035被裁到初始最大容量206.624662 kW，39步在线老化后可用上界降至206.624123 kW，造成0.00054 kW的无解边界。现统一预留初始FC容量1%，并有回归测试。

## 5. 已完成实验及正确解读

### 5.1 10种子统一框架基准（提交`bd117d7`）

路径：`data/results/testbed_multiseed/`

设置：双堆，10个种子，60 s主负载+90 s共同SOC恢复，Average与Beam；确定性健康均值。

结果：两类负载、两策略均为100%末端SOC公平、100%零硬约束违规。

同种子配对：

- 实测块：Beam主段期望退化差`-0.000334`个百分点，约`-7.49%`，95% CI `±0.000424`，8/10改善；区间跨0。
- 合成负载：差`-0.000050`个百分点，约`-1.08%`，95% CI `±0.000061`，8/10改善；区间跨0。
- 合成负载SOC等值氢耗约`-0.99%`，95% CI不跨0。
- 合成负载电池吞吐约`-28.03%`，10/10改善。

结论：框架可以公平比较；尚不能声称Beam稳定延寿。

### 5.2 长时Gamma敏感性

路径：`data/results/testbed_gamma_long/`

1000 h等效暴露、5000 Monte Carlo、终点CV 5%/10%/20%。

- CV扩大分布宽度，但没有改变已测场景排序；
- 当前事件型确定性损伤占总期望损伤约96.9%-98.2%；
- 策略排序主要受启停/变载事件系数和事件次数控制；
- 一个合成种子中Beam退化更高。

结论：Gamma适合作为不可逆在线状态和不确定性边界；短时稀疏Gamma跳变不是当前论文的核心创新。

### 5.3 固定动作75点系数网格

路径：`data/results/testbed_coefficient_sensitivity/`

扫描：连续系数`[0.5,1,2]`，启停与变载分别`[0.25,0.5,1,2,4]`，共75组合。

- 两类负载中Beam平均退化方向在75点上均没有反转；
- 95% CI明确小于0的网格比例：实测块40%，合成负载约6.7%；
- 没有任何组合做到10/10种子全部改善。

这是固定动作暴露重加权，不是闭环重新规划。它只能说明评价指标方向敏感性，不能证明控制器在改变系数后仍会选择相同或更优动作。

### 5.4 旧“全寿命损伤归一化”的闭环系数实验

路径：`data/results/testbed_closed_loop_coefficients/`

4个系数角点、5个种子、两类负载、Average/Beam，全部SOC公平、零违规。

关键失败发现：四个角点共2400个决策行的动作匹配率为100%。改变系数只改变计算出的退化值，没有改变控制动作。

原因：每秒退化增量约`10^-3%`，却除以双堆全寿命损伤参考约18.7%，使`degradation_increment`目标比氢耗、电池、SOC和性能项小几个数量级。

该实验是“旧归一化无决策影响”的失败证据，不要重复运行，也不要用它证明系数稳健。

### 5.5 新“单步损伤归一化”烟雾实验

路径：`data/results/testbed_closed_loop_step_norm_smoke/`

代码已将退化增量改为除以可审计的单步最大连续+Ramp+变载+启停损伤参考，使严重事件时归一化退化成本约为O(1)。轨迹新增各目标分量列。

4个角点、1个种子、两类负载全部SOC公平、零违规，且动作开始对退化目标响应。

但发现新的评价漏洞：实测块主段中Beam可以关闭FC、临时使用电池，在SOC恢复尾段再补能。因此：

- 主段退化可显示为0或“-100%”；
- 这不代表总任务退化为0；
- 必须把主段和恢复段相加后比较总退化、总氢耗和总电池负担。

文献基准、实测块、单个烟雾种子：

```text
Average总期望退化 = 0.004981%
Beam总期望退化    = 0.004702%
```

真实总改善约5.6%，不是主段表面显示的100%。

合成负载文献基准：

```text
Average总期望退化 = 0.004859%
Beam总期望退化    = 0.004845%
```

改善很小。变载占优角点下，合成负载Beam总退化`0.005105%`，高于Average的`0.004609%`，说明策略可发生反转。

这组实验只有1个种子，不能作论文结论。它的作用是证明新归一化会影响动作，同时暴露终端SOC/恢复尾段的策略投机问题。

## 6. 当前最重要的未完成问题

### 6.1 先修评价协议，不要马上扩展种子

当前最高优先级不是Dreamer，也不是继续扩大系数网格，而是避免控制器把退化和能量负担推迟到恢复尾段。

必须修改：

1. 主论文策略比较以“主段+SOC恢复段总指标”为主；主段指标只做机制解释。
2. 配对统计默认加入：
   - `expected_damage_increment_pct`总退化；
   - `hydrogen_soc_corrected_g`；
   - 总电池吞吐；
   - 总启停/变载次数；
   - 末端SOC误差和恢复时长。
3. 探索把末端SOC约束直接纳入规划：硬终端带、终端价值函数或足够长滚动时域，而不是完全依赖事后恢复控制器。
4. 电池使用不能只看SOC；至少保留吞吐成本，后续应加入可解释的电池退化代理。
5. 增加“FC全关+电池承担主段”的检测指标和最大连续关闭时长。

### 6.2 退化目标需要消融

新单步归一化能改变动作，但可能过强。后续必须在相同负载上比较：

- 旧全寿命归一化（仅作为失败对照）；
- 新单步归一化；
- `w_degradation = 0/0.25/0.5/1.0`；
- 当前健康性能损失项开/关；
- 只用动作增量退化、只用当前健康性能损失、两者同时使用；
- 不同末端SOC处理。

选择权重不能靠单个“好看结果”，要基于Pareto前沿、约束满足和多种子稳定性。

### 6.3 真实观测修正仍缺数据链

当前Gamma只完成：

```text
prior D(k) + action → predicted D(k+1), theta(k+1)
```

尚未完成：

```text
new voltage/current/temperature/pressure observation
→ UKF/PF/FFRLS correction
→ posterior D/theta
```

需要用户补充刘占伟MAT观测链：`data_mark_update/a/b/bestX/V/v_test`等，才能复现真实UKF-PF修正。没有这些文件时，可以先定义observer接口和仿真观测测试，但不得声称完成真实在线估计。

## 7. 预测模块当前判断

现有默认需求预测对象是整车需求功率`P_dem`，不是控制器决定的FC输出功率。

仓库已有State-Aware Direct Power Prediction：历史车速、加速度、功率、SOC等直接预测H=1/3/5/10 s需求功率；冻结默认是XGBoost。

事件条件化+概率预测候选位于：

- `src/fc_power/prediction/event_conformal.py`
- `scripts/21_event_probabilistic_power_prediction.py`
- `scripts/22_evaluate_prediction_control_regret.py`
- `data/results/prediction_event/`

已知结果：H=5/10 MAE约改善0.92%/1.80%，高功率F1小幅提高；90%区间仍欠覆盖；事件窗口下游控制有部分收益，但p05制动场景过于保守。

结论：保留为候选，不替换默认XGBoost。完成评价协议和退化目标消融后，再比较：Persistence、RLS/AR、XGBoost、事件多专家、物理+残差、TCN/GRU/PatchTST、分位数/场景预测。最终以控制收益和事件指标判断，而不是只看总体RMSE。

## 8. 论文阅读优先级和用途

PDF不应提交到Git。Codex CLI本地需单独提供论文目录。当前网页版附件路径不会自动出现在用户本机。

### 第一优先级：直接决定当前模型和评价

1. `Q-learning based energy management strategy for a hybrid multi-stack fuel cell system considering degradation.pdf`
   - Ghaderi 2023；核对式(2)、Table 4、低载/高载/自然运行/启停/变载定义；
   - 当前文献退化系数和多堆EMS最直接来源。
2. `A deterioration-aware energy management strategy for the lifetime improvement of a multi-stack fuel cell system.pdf`
   - Zuo；随机动态负载、多堆异质性、寿命均衡和评价思路；
   - 用于检查当前随机负载、事件计数和多堆分配是否合理。
3. `An Energy Management Strategy for Multistack Fuel Cell Hybrid Locomotives With Integrated Optimization of System Service Life.pdf`
   - 多堆服务寿命联合优化和负载分配基准。
4. `多模块燃料电池系统效率优化及功率调控策略研究--陈鹏.pdf`
   - 国内多模块功率调控结构、仿真负载来源和指标表达；
   - 可作为工程结构参考，不直接照搬结论。

### 第二优先级：健康估计、RUL和维护

5. `基于在线辨识和极小值原理的PEMFC混合动力系统综合能量管理方法.pdf`
   - 在线参数辨识与能量管理耦合；用于设计observer接口。
6. `Remaining useful life prediction ... Wiener process and Bayesian GRU ...pdf`
   - Wiener/贝叶斯GRU和多源不确定性；适合大论文健康预测升级，不替代当前Gamma动作转移。
7. `Reinforcement learning-based maintenance scheduling ... stack-to-stack heterogeneity.pdf`
   - 随机退化、堆间异质性、维护调度；用于后期寿命/维护扩展。
8. `Optimal Post-Prognostics Decision Making ... Joint Load Allocation and Maintenance Scheduling.pdf`
   - 负载分配与维护联合决策；属于长期扩展，不是当前第一阶段。

### 第三优先级：生命周期和空供优化

9. `Adaptive optimization strategy of air supply for automotive PEMFC in life cycle.pdf`
10. `A quick evaluating method for automotive fuel cell lifetime.pdf`
11. `Lifetime prediction and the economic lifetime of PEMFCs.pdf`

用于补充寿命尺度、经济寿命、生命周期参数自适应，不应与当前多堆功率调控主线混为一个实验。

### 高级世界模型论文

12. `TD-MPC.pdf`
   - TD-MPC2是decoder-free潜在动力学+奖励/Q头+策略先验+MPC局部轨迹优化；
   - 原论文主要面向连续动作。当前离散多堆档位不能直接照搬连续采样规划；可借用潜在动力学和TD价值学习，动作仍用离散Beam或安全投影；
   - 网页版当前副本曾出现PDF XRef损坏，Codex CLI应使用干净原PDF。
13. `EAWM.pdf`
   - EAWM是事件预测和Generic Event Segmentor辅助世界模型表示学习，不是独立控制器；
   - PEMFC第一版应直接使用物理事件：启停、变载、高载、制动、SOC紧张、健康失衡，不照搬视觉AGMM。
14. `dreamer25.pdf`
   - 网页版对话显示该文件可用，但当前scratch没有实际副本；迁移到Codex CLI前需用户把原PDF放到本地论文目录；
   - 重点核对RSSM、离散/连续latent、imagined rollout、actor-critic目标和训练稳定性。

## 9. 高级算法适配判断

### 9.1 推荐主线

```text
机理世界模型（当前）
→ 学习一步残差
→ 验证未见片段的多步rollout
→ Dreamer-style RSSM
→ EAWM事件辅助头/GES
→ actor候选动作或latent MPC
→ 安全投影
→ 统一测试框架评价
```

### 9.2 为什么不能直接上Dreamer/EAWM

- 当前刚发现退化目标旧归一化完全不影响动作；
- 新归一化又会诱导FC主段关闭、将成本转移到恢复尾段；
- 如果此时训练高级RL，算法只会更高效地利用评价漏洞；
- 先稳定目标、约束、终端SOC和总任务指标，学习算法才有意义。

### 9.3 TD-MPC的定位

- 当前动作是每堆离散电流档位+开关，双堆81组合、三堆729组合；
- TD-MPC2连续CEM式规划不直接适配；
- 若后续把各堆功率放松为连续动作，可做TD-MPC式对照；
- 当前更适合Dreamer潜在模型+离散策略头，或latent dynamics+离散Beam。

### 9.4 GPU使用

网页版执行环境没有可用PyTorch/GPU；用户电脑有PyTorch和GPU。Codex CLI接手后先运行：

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
PY
```

P0-P4机理、测试和XGBoost工作不依赖GPU。残差神经网络、Dreamer/EAWM训练再启用GPU。

## 10. 后续执行计划

### 阶段A：修正公平评价协议（最高优先级）

1. 修改`paired_strategy_comparison`和报告，默认比较总退化而非只比较主段退化。
2. 增加总启停/变载、主段FC关闭比例、最大连续关闭、恢复段长度/能量/损伤。
3. 设计至少两种末端SOC处理：
   - 现有共同恢复尾段；
   - 规划内终端SOC硬带或终端价值。
4. 做小规模单种子验证，确认策略不能靠推迟成本得到虚假收益。
5. 通过后再跑5-10种子；不要先跑30种子。

验收：100%零硬违规；SOC口径公平；主段和总指标不矛盾；结果不依赖隐藏恢复成本。

### 阶段B：退化目标和权重消融

1. 旧/新归一化对照；
2. `w_degradation`网格；
3. 性能损失项开/关；
4. 系数角点闭环重新规划；
5. 以Pareto前沿和配对CI选候选，不预设“最优固定权重”。

验收：改变退化权重会改变动作；经济性、寿命、电池负担的权衡可解释；多种子方向稳定。

### 阶段C：真实健康观测接口

1. 定义`HealthObserver.predict/correct`协议；
2. 无MAT时用合成观测测试接口；
3. 获得MAT后复现UKF-PF；
4. 记录prior/posterior，不覆盖真实观测与模型预测的边界。

### 阶段D：未来需求预测回归

在统一评价协议上重新比较事件条件化、概率场景和现有XGBoost。只有下游氢耗/退化/SOC/切换稳定改善时升级默认预测器。

### 阶段E：残差世界模型

输入当前机理状态、动作和事件，学习真实下一状态与机理预测之间的残差。先比较一步和多步rollout误差，必须在未见连续运行片段上评价。

### 阶段F：Dreamer/EAWM

推荐状态：

```text
P_dem历史/预测、SOC、P_bat、上一动作、驻留时间、
每堆D/theta/SOH/I/V/P/on、健康差异、事件标签
```

动作：离散组合动作ID或每堆factorized categorical；执行前必须安全投影。

EAWM事件：启停、变载、高载、怠速、制动、SOC紧张、健康失衡。

基线：Average、Rotating、Instant、机理Beam、预测MPC、Perfect-preview。必须做事件头、Gamma状态、latent模型和安全层消融。

## 11. 明确禁止的错误结论

- 不得把`C_deg(I|theta)`称为材料退化直接观测；它是当前健康下的性能损失代理。
- 不得声称Gamma方差由单条LZW轨迹可靠辨识；CV是敏感性假设。
- 不得把预测态`D/theta`称为已完成观测修正的posterior。
- 不得用合成半马尔可夫负载代表真实车辆分布。
- 不得只报告主段退化并隐藏SOC恢复段。
- 不得引用单种子“-100%主段退化”作为延寿结论。
- 不得在评价协议未修好时训练Dreamer/EAWM并报告收益。
- 不得混用Pei/Ghaderi、其他论文或人为调出的退化系数而不记录来源。

## 12. 关键文件索引

| 作用 | 文件/目录 |
|---|---|
| 总体进度与决策日志 | `docs/project_execution_tracker.md` |
| 本交接文档 | `docs/CODEX_CLI_HANDOFF_2026-07-12.md` |
| Gamma状态转移 | `src/fc_power/health/gamma_process.py` |
| 文献系数、LZW标定、theta映射 | `src/fc_power/health/lzw_gamma_calibration.py` |
| 动态性能损失 | `src/fc_power/health/dynamic_proxy.py` |
| 多堆机理世界模型 | `src/fc_power/world_model/mechanistic.py` |
| LZW世界模型工厂 | `src/fc_power/world_model/lzw_factory.py` |
| 多堆规划器/安全投影 | `src/fc_power/power_allocation/multistack_allocator.py` |
| 随机/真实块负载 | `src/fc_power/evaluation/load_profiles.py` |
| 统一测试执行器 | `src/fc_power/evaluation/multistack_testbed.py` |
| 系数暴露审计 | `src/fc_power/evaluation/degradation_sensitivity.py` |
| 统一测试入口 | `scripts/23_run_multistack_testbed.py` |
| 长时Gamma敏感性 | `scripts/24_long_horizon_gamma_sensitivity.py` |
| 系数网格敏感性 | `scripts/25_degradation_coefficient_sensitivity.py` |
| 10种子主结果 | `data/results/testbed_multiseed/` |
| 长时Gamma结果 | `data/results/testbed_gamma_long/` |
| 旧归一化失败证据 | `data/results/testbed_closed_loop_coefficients/` |
| 新归一化烟雾结果 | `data/results/testbed_closed_loop_step_norm_smoke/` |

## 13. 可直接粘贴给 Codex CLI 的启动提示

```text
你正在接手xxluestc/FC项目。先完整阅读：
1) docs/CODEX_CLI_HANDOFF_2026-07-12.md
2) docs/project_execution_tracker.md

先检查git分支、HEAD、status和最近提交，禁止reset/checkout覆盖已有修改。
确认至少包含提交1b8e8de，并运行61项测试、baseline check、compileall和git diff --check。

当前最高优先级是修正公平评价协议：所有主结论必须比较主段+SOC恢复段的总退化、总氢耗和总电池负担；增加FC关闭比例、最大连续关闭、启停/变载次数和恢复成本；比较恢复尾段与规划内终端SOC约束，防止策略推迟成本。

先做单种子小实验验证协议，再做退化归一化/权重/性能损失项消融，最后扩展多种子。不要重复旧全寿命归一化四角点实验；它已经证明2400行动作100%相同。不要直接训练Dreamer/EAWM，直到目标、约束和评价口径通过。

保护data/key/li_junhao_dual_stack/engine5_three_day_processed.csv.gz，不暂存、不还原。大型轨迹不提交。每完成一个阶段，更新project_execution_tracker，运行测试，做独立commit并生成format-patch。
```
