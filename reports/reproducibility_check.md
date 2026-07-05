# 可复现性检查

检查日期：2026-07-05。

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

## 关键复现约束

- 原始大文件不提交Git，需通过本地路径配置和编号脚本重建。
- 模型按70/15/15时间顺序切分并设置H秒purge，不随机打乱。
- 最终模型只按验证集分数选择；测试集不参与选模。
- 权重只在控制测试序列前1200秒搜索，完整3600秒用于统一对照。
- `degradation proxy`不是已验证的真实材料退化系数。

## 本轮修复

Windows下并行MultiOutput HistGradientBoosting曾产生过量子进程；已改为顺序多输出头、固定迭代数和确定性训练子样本。H=1回归输出的一维/二维差异也已统一，所有Python文件经Black格式化并通过compileall。
