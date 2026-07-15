# Codex CLI 交接：陈鹏曲线口径与 MTS 功率分配路线

更新时间：2026-07-15

## 1. 当前仓库状态

```text
分支：codex/gamma-health-foundation
本次工作起点：8d8475b
验证：159项 unittest 全部通过
baseline：轻量检查通过
compileall：通过
git diff --check：通过，仅有 Windows LF/CRLF 提示
```

工作区原有未跟踪内容如下，本次没有修改、删除或纳入提交：

```text
data/results/empirical_event_load/
scripts/71_fit_empirical_event_load.py
```

## 2. 本次解决了什么

### 2.1 陈鹏三条曲线的身份已经查清

通过陈鹏论文、MATLAB 源码和 Origin 原始工程三方核对，确认：

```text
第四章曲线横轴 = 电堆毛功率
第四章曲线效率 = 扣除空压机后的 LHV 系统效率
论文第三章效率公式 = HHV
```

所以原结果存在毛/净功率和 LHV/HHV 表述不统一。后续功率平衡统一使用净功率，效率统一使用 LHV。

Origin `Book7/Sheet5` 的 69 个点已进入仓库，按陈鹏原代码常数重建净功率。三条曲线的端点只能称为数据插值域，不能称为已经验证的物理最小、最大功率。

### 2.2 当前算法路线已经收窄

当前主线是：

```text
当前净功率需求
-> 枚举最多两堆运行的可行组合
-> 每个组合内部连续优化净功率分配，得到当前氢耗服务代价
-> 外层在线选择运行组合，同时考虑启停切换代价
-> 执行动作并进入下一秒
```

文献分工如下：

```text
Chen：异构效率曲线和瞬时效率目标
Igourzal 2024：运行组合枚举、组合内优化、故障重构
Borodin 1992：当前代价已知、未来未知的 MTS 外层状态选择
NeurIPS 2021：连续功率的 hitting cost + movement cost 思路
Haubensak 2026：启停二进制变量和氢耗-启停权衡
AAAI 2023：未来预测扩展，当前不使用
Automatica 2025：混合离散连续问题分类，当前不直接使用其 bandit 算法
```

重要修正：Borodin 1992 原文算法是 nearly-oblivious `A_f`，不是文中所谓的 WFA。后续不得把普通累计动态规划或 one-step greedy 命名为 WFA。

## 3. 新增和修改文件

```text
data/upstream_chen/chen_efficiency_curves_origin_sheet5.csv
data/upstream_chen/README.md
data/processed/chen_efficiency_curves_audited.csv
data/processed/chen_efficiency_curves_audit.json
data/results/chen_efficiency_curve_audit/fig35_chen_curve_basis_audit.png
src/fc_power/power_allocation/chen_efficiency_curves.py
scripts/74_build_chen_efficiency_curve_audit.py
tests/test_chen_efficiency_curves.py
docs/CHEN_EFFICIENCY_CURVE_AND_ONLINE_ALGORITHM_AUDIT_2026-07-15.md
docs/project_execution_tracker.md
.gitignore
```

主审计文档：

```text
docs/CHEN_EFFICIENCY_CURVE_AND_ONLINE_ALGORITHM_AUDIT_2026-07-15.md
```

## 4. 本地文献和证据路径

陈鹏源码：

```text
G:/大论文/2025陈鹏/论文/仿真模型/cal_eff.m
G:/大论文/2025陈鹏/论文/仿真模型/get_condition.m
G:/大论文/2025陈鹏/论文/仿真模型/Untitled3_6.m
G:/大论文/2025陈鹏/论文/实验数据/obsidian/第四章.opju
```

本轮使用的参考文献目录：

```text
G:/大论文/AI文献库/陈鹏-效率优化的功率调控-相关参考文献
```

该目录现已包含 Borodin 1992、NeurIPS 2021、AAAI 2023、Automatica 2025 和两篇 ECM 多堆论文。当前无需用户继续下载或打开文献。

## 5. 复现命令

```powershell
$env:PYTHONPATH='src'
python scripts/74_build_chen_efficiency_curve_audit.py
python -m unittest discover -s tests -v
python scripts/run_baseline.py --mode check
python -m compileall -q src scripts tests
git diff --check
```

## 6. 下一步唯一优先任务

先实现陈鹏净功率/LHV曲线上的组合内连续求解器，不直接实现复杂在线算法。该求解器必须对每个需求点和运行组合返回：

```text
各堆净功率
各堆毛功率
各堆 LHV 效率
总氢耗或 LHV 化学输入
功率平衡误差
是否超出曲线插值域
```

然后按顺序实现和比较：

```text
Chen instantaneous
Igourzal sticky configuration
one-step greedy with switching cost
offline dynamic programming oracle
Borodin discrete-time A_f
```

在前四项的成本和约束闭环没有验证前，不进入 `A_f`，也不加入退化、电池或未来需求预测。

## 7. 当前仍缺但不阻塞的物理参数

```text
厂家额定净功率
物理最小稳定净功率
最大爬坡速率
启动和停机的真实成本是否不同
```

第一轮可以严格限制在陈鹏曲线数据域内，并把切换权重作为无量纲灵敏度参数。准备物理可信度实验和论文定量结论前，必须补齐上述参数或明确引用来源。
