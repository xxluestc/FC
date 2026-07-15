# Codex CLI 交接：陈鹏动态功率分配基础闭环

更新时间：2026-07-15

## 仓库状态

```text
分支：codex/gamma-health-foundation
原始动态基础检查点：60d1495（N+1容量口径已被后续提交纠正）
容量纠错与故障结果检查点：46ec207
测试：174 项 unittest 全部通过
baseline：轻量检查通过
compileall：通过
git diff --check：通过，仅有 Windows LF/CRLF 提示
```

以下是用户原有未跟踪内容，本轮没有修改、删除或纳入提交：

```text
data/results/empirical_event_load/
scripts/71_fit_empirical_event_load.py
```

## 当前结论

陈鹏三条曲线已统一为净功率横轴和 LHV 效率。三堆正常最多两堆运行，组合内采用分段线性断点枚举得到全局最优分配，组合外比较 Average、Daisy、Instantaneous、Sticky、One-step greedy、Break-even hysteresis 和 Offline DP。

10 个开发种子和 10 个独立留出种子均跑通。留出集 online hysteresis 相对 instantaneous：

```text
氢耗：+0.210% ± 0.049%
状态变化：-46.588% ± 11.528%
功率平衡最大绝对误差：< 1e-8 kW
```

基准切换权重下综合目标尚未超过 instantaneous；权重为 `2x` 时，留出集综合目标改善 `0.159% ± 0.042%`，开发集改善 `0.156% ± 0.039%`。必须按 Pareto 和敏感性报告，不能只报有利权重。

## 关键文件

```text
src/fc_power/power_allocation/chen_efficiency_curves.py
src/fc_power/power_allocation/chen_dispatch.py
src/fc_power/power_allocation/chen_dispatch_policies.py
src/fc_power/evaluation/chen_dynamic_load.py
scripts/74_build_chen_efficiency_curve_audit.py
scripts/75_run_chen_dynamic_dispatch_foundation.py
scripts/76_plot_chen_dynamic_dispatch_results.py
scripts/77_run_chen_n_plus_one_fault_reconfiguration.py
scripts/78_plot_chen_n_plus_one_fault_results.py
tests/test_chen_efficiency_curves.py
tests/test_chen_dispatch.py
tests/test_chen_dispatch_policies.py
tests/test_chen_dynamic_load.py
docs/CHEN_DYNAMIC_DISPATCH_FOUNDATION_RESULTS_2026-07-15.md
docs/CHEN_N_PLUS_ONE_FAULT_RESULTS_2026-07-15.md
```

结果目录：

```text
data/results/chen_dynamic_dispatch_foundation/
data/results/chen_dynamic_dispatch_holdout/
data/results/chen_n_plus_one_fault_reconfiguration/
```

图：

```text
data/results/chen_dynamic_dispatch_holdout/figures/fig36_chen_dynamic_dispatch_trajectory.png
data/results/chen_dynamic_dispatch_holdout/figures/fig37_chen_dynamic_dispatch_tradeoff.png
```

## 重要边界

- `340 cells / 120 kW` 来自陈鹏论文测试平台；按片数推断的 `95.29/105.88/116.47 kW` 毛功率只作背景，不用于调度。
- 当前有效净功率域为曲线插值域 `5.664-54.263 / 6.635-60.079 / 7.607-65.807 kW`，不能称为厂家物理边界。
- Zuo fast 矩阵只控制随机事件顺序，负载幅值完全由陈鹏曲线推导。
- 正常最强两堆上限为 `125.885 kW`，任一单堆故障后的保证容量为 `114.342 kW`；两者不得再混称。
- 切换权重是等效氢耗权衡参数，不是退化系数或真实维护成本。
- 当前不含退化、电池、未来需求、真实爬坡和最小驻留；故障验证假设故障当步被完美识别和隔离。
- Offline DP 使用完整未来序列，只是事后下界。
- One-step greedy 与 Sticky 在当前主权重下行为相同，不得包装成独立改进。

## 复现命令

```powershell
$env:PYTHONPATH='src'
python scripts/74_build_chen_efficiency_curve_audit.py
python scripts/75_run_chen_dynamic_dispatch_foundation.py --split-label development --out-dir data/results/chen_dynamic_dispatch_foundation
python scripts/75_run_chen_dynamic_dispatch_foundation.py --seeds 3026 3027 3028 3029 3030 3031 3032 3033 3034 3035 --split-label holdout --out-dir data/results/chen_dynamic_dispatch_holdout
python scripts/76_plot_chen_dynamic_dispatch_results.py
python scripts/77_run_chen_n_plus_one_fault_reconfiguration.py
python scripts/78_plot_chen_n_plus_one_fault_results.py
python -m unittest discover -s tests -v
python scripts/run_baseline.py --mode check
python -m compileall -q src scripts tests
git diff --check
```

## 下一步唯一优先任务

单堆永久故障退出和 N+1 在线重构已经完成。下一步做未知爬坡率与最小驻留时间的参数敏感性，并明确区分“扫描边界”和“厂家真实参数”；随后用未参与构造的实车净功率轨迹回放。

不要回到 Gamma 退化路线，也不要先加入锂电池或未来需求预测。
