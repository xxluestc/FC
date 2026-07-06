# 可复现性检查

检查日期：2026-07-05。

## 冻结baseline检查（2026-07-06）

`python scripts/run_baseline.py --mode run` 已从现有canonical实际完成：动力学/需求功率、固定XGBoost H=1/3/5/10、固定权重四策略和汇总绘图。总运行约460秒，其中预测训练约112秒。该入口没有调用07--12优化脚本。

`python scripts/run_baseline.py --mode check` 已通过；它只检查Git提交的小型CSV/JSON，不依赖原始CSV、MAT或本地绝对路径。`python scripts/06_run_all_experiments.py`也增加了baseline四策略和schema检查。

依赖本地大数据的步骤：从原始CSV重建canonical、由canonical重建约70 MB需求表、训练预测并写入约数十MB预测表、写入逐秒分配轨迹。上述大文件被`.gitignore`排除；仓库只提交冻结配置、代码、小型指标、报告和PNG。

## 执行结果

以下命令均已实际通过：

```powershell
python -m black src scripts
python -m compileall src scripts
python scripts/04_train_or_run_predictors.py ...
python scripts/05_run_power_allocation.py ...
python scripts/plot_results.py
python scripts/06_run_all_experiments.py
python scripts/06_run_all_experiments.py --full-local
```

预测全流程约250秒；MPC权重搜索、四策略和诊断约196秒。生成结果通过schema、四模型族、H=1/3/5/10完整性和全部策略末端SOC ±0.02检查；本次最大绝对末端SOC偏差为0.00193。

新增优化链已实际运行：字段组消融约788秒；两个较早完整运行段上的48组鲁棒控制搜索约716秒；7个Pareto候选晚期泛化约45秒。最终测试使用更晚的独立3600秒连续运行段，默认优化Predicted末端SOC误差为-0.01036。其他基线存在明显SOC亏损，因此报告同时生成SOC等值补能指标，未将raw成本直接冒充公平比较。

## 关键复现约束

- 原始大文件不提交Git，需通过本地路径配置和编号脚本重建。
- 模型按70/15/15时间顺序切分并设置H秒purge，不随机打乱。
- 最终模型只按验证集分数选择；测试集不参与选模。
- 权重只在控制测试序列前1200秒搜索，完整3600秒用于统一对照。
- `degradation proxy`不是已验证的真实材料退化系数。
- 当前晚期控制段需求裁剪比例为5.94%，结论限定为可行域内测试。

## 本轮修复

Windows下并行MultiOutput HistGradientBoosting曾产生过量子进程；已改为顺序多输出头、固定迭代数和确定性训练子样本。H=1回归输出的一维/二维差异也已统一，所有Python文件经Black格式化并通过compileall。
