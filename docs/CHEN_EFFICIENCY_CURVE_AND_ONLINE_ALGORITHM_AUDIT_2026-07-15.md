# 陈鹏效率曲线口径审计与在线功率分配路线

更新时间：2026-07-15

## 1. 三个问题的直接答案

### 1.1 功率是净功率还是电堆毛功率

陈鹏第四章保存的三条曲线横轴是**电堆毛功率**，不是系统净功率。

证据来自 MATLAB 源码：

- `get_condition.m:73` 返回 `V * I * cell_count * 406`，没有扣除附件功耗；
- `cal_eff.m:9` 把这项记为 `power_stack`；
- `cal_eff.m:39` 另行计算 `sim_net_power = power_stack - sim_all_cmp_power`。

因此原曲线存在一个口径混用：横轴是毛功率，效率分子却是扣除空压机后的净功率。后续若把需求定义为直流母线需要燃料电池系统提供的功率，应统一使用本次重建的 `net_system_power_kw`，不能直接拿原始横轴做功率平衡。

### 1.2 效率采用 LHV 还是 HHV

**第四章实际结果数据采用 LHV。论文第三章公式采用 HHV，两者不一致。**

证据如下：

- 论文第三章式 (3-4) 使用 `eta_HHV` 和等效电压 `1.482 V`；
- `cal_eff.m:40` 由法拉第定律计算氢流量；
- `cal_eff.m:43` 使用 `sim_net_power / H2_flow / 1200000` 输出百分数。氢流量单位为 kg/s，`1200000 = 120 MJ/kg / 100`，所以这是 LHV 百分效率。

当前路线固定为：**净功率横轴 + LHV 效率**。若仅为展示需要转换为 HHV，可统一乘以 `241.98 / 286.02`；统一换算不会改变同一时刻的效率排序，但会改变效率和成本的数值大小。

### 1.3 三条曲线是否有最小、最大输出范围

有**数据采样范围**，但没有证据证明它们就是电堆的物理最小、最大输出限制。

`Untitled3_6.m` 使用 `0.1:0.05:1.2 A/cm2`，原始 Origin 工程 `Book7/Sheet5` 保存了每堆 23 个点。按当前统一口径得到：

| 虚拟堆 | 电池片数 | 原始毛功率范围 (kW) | 重建净功率范围 (kW) | LHV 峰值效率 | 峰值对应电流密度 (A/cm2) |
|---|---:|---:|---:|---:|---:|
| stack 1 | 270 | 8.743-62.085 | 5.664-54.263 | 48.074% | 0.35 |
| stack 2 | 300 | 9.714-69.959 | 6.635-60.079 | 48.635% | 0.35 |
| stack 3 | 330 | 10.686-78.456 | 7.607-65.807 | 49.158% | 0.30 |

不能把表中端点写成额定物理边界，原因是：

- 空压机喘振和阻塞检查在 `cal_eff.m` 中被注释；
- `optimize_with_OOA.m` 虽然接收外部最小、最大功率，但现有目录没有找到调用者或参数来源；
- 原始曲线不包含关闭状态，关闭必须作为独立离散状态建模。

现阶段可以把表中范围称为“**陈鹏模型有数据支持的插值域**”。超出范围不外推；物理额定范围以后若取得厂家或台架参数，再替换约束即可。

## 2. 本次已经固化的数据

原始 Origin 快照：

```text
data/upstream_chen/chen_efficiency_curves_origin_sheet5.csv
```

可复算脚本：

```text
PYTHONPATH=src python scripts/74_build_chen_efficiency_curve_audit.py
```

生成结果：

```text
data/processed/chen_efficiency_curves_audited.csv
data/processed/chen_efficiency_curves_audit.json
data/results/chen_efficiency_curve_audit/fig35_chen_curve_basis_audit.png
```

每个点按陈鹏代码使用的常数重建：

```text
I_j = current_density_j * 406
mH2_j = I_j * 2.02e-3 * cell_count_j / (2 * 96485)
Pchem,LHV_j = mH2_j * 120e6
Pnet_j = eta_LHV_j * Pchem,LHV_j
Paux_j = Pgross_j - Pnet_j
```

这不是新拟合，也没有改动陈鹏的效率值，只是把原代码中隐含的净功率还原出来。

## 3. 当前论文框架如何由已有文献组合

当前不直接套某一个“更高级”的算法名字，而是让每一层都对应一篇已有工作。

| 组成部分 | 采用的论文依据 | 当前借用什么 | 当前不借用什么 |
|---|---|---|---|
| 异构效率曲线和瞬时效率目标 | 陈鹏学位论文 | 三堆异构效率曲线、氢耗/效率目标 | 原 OOA 和混用的功率口径 |
| 运行组合枚举和组合内功率优化 | Igourzal et al., ECM 2024 | 枚举可行堆组合、组合内连续优化、故障组合剔除 | 其无法由本项目数据验证的真实老化成本 |
| 离散运行组合与切换代价 | Borodin et al., JACM 1992 | Metrical Task System 的状态、当前服务代价和状态转移代价 | 不把普通动态规划冒充 Work Function Algorithm |
| 连续功率变化平滑 | Zhang et al., NeurIPS 2021, *Revisiting Smoothed Online Learning* | 当前代价已知时，联合当前代价与 movement cost 的思路 | 未验证凸性前，不声称继承其理论界 |
| 启停二进制变量和启停惩罚 | Haubensak et al., ECM 2026 | 启停变量、切换计数和效率/启停权衡的实验设计 | 未来负载、温度预测和 MI-MPC 暂不加入 |
| 未来预测扩展 | Lin et al., AAAI 2023 | 后续可加入预测不确定性和动态规划窗口 | 当前阶段明确不用未来需求预测 |
| 混合决策问题分类 | Chi et al., Automatica 2025 | 离散堆集合 + 连续功率的形式化描述 | 不直接使用 Exp3.S/带随机探索算法 |

Automatica 2025 的算法当前不适合直接照搬。它面向未知奖励、bandit 反馈和随机探索，并要求连续域凸、收益凹且离散集合收益单调不减。当前三条效率曲线已知，需求在决策前已知，增加运行堆数也不保证效率单调提高。直接使用会违反其关键假设，还会引入没有必要的在线随机探索。

AAAI 2023 的主算法依赖未来代价预测。当前明确不做未来功率预测，所以只把它保留为第二阶段扩展依据。

## 4. 当前应实现的基础算法

### 4.1 离散状态

正常最多两堆运行，基础状态集合为：

```text
M = {1, 2, 3, 12, 13, 23}
```

零需求时另设 `OFF`。故障堆所在组合直接判为不可行。若加入最小驻留时间，需要把剩余驻留计数并入状态，不能只在目标函数里随意加一项。

### 4.2 每个组合的当前服务代价

当步需求 `Pdem,t` 已知后，对每个可行组合 `m` 解一个很小的连续问题：

```text
c_t(m) = min sum_j Pchem,LHV,j(Pj)

subject to
sum_j Pj = Pdem,t
Pj in Chen net-power interpolation domain
only stacks in m may produce power
fault and hard ramp constraints are satisfied
```

两堆运行时只有一个独立连续变量，可以用一维网格加局部精化得到可审计的近似全局最优解，不需要先上黑箱强化学习。

### 4.3 外层切换选择

外层把每个运行组合看成一个状态，把 `c_t(m)` 看成当前服务代价，把启动/停机看成状态转移代价。这正是 MTS 的问题结构：先看到当前代价，再选择状态，未来代价未知。

基础试验先用对称切换距离：

```text
d(m_prev, m) = lambda_switch * number_of_changed_stack_states
```

这里的 `lambda_switch` 是运行策略的切换权重，不宣称是真实退化系数。做灵敏度试验后报告“氢耗-切换次数”的 Pareto 关系。若将来取得启动和停机不同的实际成本，则转为一般 task system，而不再声称转移代价是对称度量。

Borodin 1992 原文给出的最优确定性 MTS 算法是基于累计任务代价阈值的 nearly-oblivious 算法 `A_f`。它不是文中所谓的 WFA。本项目下一步应实现并验证 `A_f` 的离散时间版本，同时保留以下容易解释的基线：

```text
Equal distribution
Daisy chain
Chen instantaneous optimizer
Igourzal sticky configuration: previous combination feasible时保持
one-step greedy: current service cost + switching cost
Borodin MTS A_f: proposed online outer policy
offline dynamic programming: only作事后下界，不可在线使用
```

### 4.4 连续功率平滑

在已选组合内部，再比较：

```text
instantaneous efficiency optimum
vs.
efficiency cost + gamma * |P_t - P_{t-1}|
```

这一项对应 NeurIPS 2021 的 hitting cost + movement cost 思路。由于陈鹏插值曲线是否满足论文要求的凸性和 quadratic growth 尚未检验，当前只能引用问题结构和算法思想，不能照搬其竞争比或动态遗憾结论。凸性检查是进入理论包装前的硬门槛。

## 5. 论文可讲的核心问题

当前小论文不再把重点放在“换一个优化器名字”，而是解决下面这个完整问题：

> 在当前净功率需求已知、未来需求未知的条件下，面向异构三堆 N+1 系统，联合选择运行堆组合和组合内净功率分配，在满足功率、故障、启停和爬坡约束的同时，减少氢耗与不必要切换。

有效性必须通过以下结果证明：

- 相对 Equal、Daisy、Chen instantaneous 的氢耗和平均效率；
- 相对 Chen instantaneous 的启停/换堆次数；
- 相对 offline DP 的总成本差距；
- 故障注入后是否继续满足功率；
- 不同切换权重下的氢耗-切换 Pareto 曲线；
- 在实车负载回放和随机动态负载上的一致结论。

在这些基础结果跑通前，不加入电池、未来需求预测、Gamma 退化或复杂世界模型。

## 6. 当前边界和下一步

目前已经解决的是输入口径和算法归属，尚未证明新策略优于基线。下一步唯一主任务是：

1. 用审计后的净功率/LHV 曲线实现组合内连续求解器；
2. 先跑瞬时优化、sticky 和 offline DP，确认功率平衡及成本定义；
3. 再按 Borodin 1992 实现离散时间 `A_f`，与 one-step greedy 比较；
4. 只有出现稳定的氢耗-切换改进后，再讨论算法升级或论文表述。

当前不需要用户补充陈鹏数据。厂家额定功率、物理最小稳定功率和爬坡限制仍然未知，但不阻塞第一轮“数据支持域内”的算法链验证；这些参数在准备物理可信度实验前必须补齐。

## 7. 2026-07-15 动态闭环结果更新

本文件原计划中的组合内全局求解、在线基线、offline DP、10 个开发种子和 10 个独立留出种子已经完成。主结果、负载构造、每步决策链、图和限制统一记录在：

```text
docs/CHEN_DYNAMIC_DISPATCH_FOUNDATION_RESULTS_2026-07-15.md
```

留出集 online hysteresis 相对 instantaneous 平均减少 `46.59%` 的电堆状态变化，氢耗增加 `0.210%`；在切换权重 `2x` 时综合目标改善 `0.159%`。当前结论是可重复的效率-切换 Pareto，不是寿命或真实退化收益。

故障注入前发现原结果把正常最强两堆上限 `125.885 kW` 误称为 N+1 保证容量。现已改为最不利单堆故障后的 `114.342 kW`，并全量重跑开发/留出结果。三种故障身份、七种策略、十个留出种子的 210 条运行均实现故障当步零缺额重构；详细结果见 `docs/CHEN_N_PLUS_ONE_FAULT_RESULTS_2026-07-15.md`。

原计划中的 Borodin `A_f` 暂不实现。下一优先级是爬坡/最小驻留敏感性；在没有厂家参数时只报告参数域，不声称真实物理约束。
