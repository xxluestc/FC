# 可复现性检查

检查日期：2026-07-05。

## 代码格式

全部 `src/**/*.py` 和 `scripts/*.py` 已运行Black格式化；不再保留压缩成单行的实现。检查命令：

```powershell
python -m black --check src scripts
```

## 编译检查

```powershell
python -m compileall -q -f src scripts
```

结果：通过，无SyntaxError或ImportError。

## 主入口检查

```powershell
python scripts/06_run_all_experiments.py
```

默认入口只检查Git提交的小型结果、关键schema、三类预测方法、H=1/3/5/10/15完整性，以及全部策略末端SOC是否位于±0.02带内，因此新克隆仓库不依赖私有原始数据即可运行。结果：通过。

本机完整中间数据检查使用：

```powershell
python scripts/06_run_all_experiments.py --full-local
```

结果：通过。

## 实验脚本实际运行

- 预测脚本：约147秒（其中向量化特征构建约1.6秒、共享树拟合约17秒，其余为窗口模型、推理、指标和CSV写入）；
- 功率分配与敏感性脚本：约131秒；
- 所有大型原始/处理中间CSV由 `.gitignore` 排除，需通过编号脚本由本地原始数据生成；
- 提交仓库包含小型指标表、配置模板、代码、报告和图，不包含token、原始MAT/CSV或本地路径配置。

## 已修复问题

1. 原代码大量单行语句：Black统一格式；
2. 特征构造每个窗口重复整列转NumPy，近似O(n²)：改为滑动窗口向量化，特征阶段降至约1.6秒；
3. 一个模型结果混称物理预测：拆为speed-only、direct-power、physics-corrected；
4. `06`原先只检查文件存在：增加schema、方法、时域和SOC约束检查；
5. 预测结果旧方法名导致画图失效：绘图脚本已同步新schema；
6. 功率分配旧脚本只比较H=10且只相对Constant：增加H=3/5/10及Instant/Constant双基线。
