# Baseline运行手册

## 1. 文件英文名与中文含义

| 文件 | 中文名 | 作用 |
|---|---|---|
| `configs/baseline.yaml` | 基线冻结配置 | 固定数据、H=5、XGBoost、约束和权重 |
| `scripts/run_baseline.py` | 运行基线主流程 | 唯一推荐入口 |
| `00_audit_data.py` | 00_审计数据源 | 只读扫描韩/刘/李目录，不复制原始数据 |
| `01_preprocess_data.py` | 01_预处理原始整车数据 | 原始日CSV→canonical |
| `02_build_stack_degradation_h2.py` | 02_构建电堆退化代理与氢耗表 | 刘占伟性能损失表→档位表 |
| `03_vehicle_dynamics_power.py` | 03_构建需求功率与动力学基线 | canonical→需求/动力学表 |
| `04_train_or_run_predictors.py` | 04_训练或运行需求预测器 | baseline固定XGBoost；多模型模式属于优化 |
| `baseline/05_run_baseline_allocation.py` | 05_运行基线功率分配 | 固定权重四策略，不搜索 |
| `baseline/06_summarize_baseline.py` | 06_汇总基线结果 | 汇总JSON和两张图 |
| `06_run_all_experiments.py` | 06_检查实验可复现性 | 检查已生成结果schema |
| `07--12` | 优化实验脚本 | baseline阶段不运行 |

## 2. 环境

```powershell
pip install -r requirements.txt
python -m black --check src scripts
python -m compileall src scripts
```

Python依赖包括numpy、pandas、scipy、scikit-learn、matplotlib、PyYAML和XGBoost。不要提交`configs/paths.local.yaml`、原始数据、token或绝对路径。

## 3. 无原始数据轻量检查

```powershell
python scripts/run_baseline.py --mode check
```

该模式只读取Git中提交的小型指标和schema，不训练、不读取大CSV。

## 4. 从已有canonical跑完整baseline

```powershell
python scripts/run_baseline.py --mode run
```

顺序为：canonical→动力学/需求功率→固定XGBoost→固定四策略→汇总。大约需要数分钟，生成的大型需求、预测和轨迹CSV被Git忽略。

## 5. 从本地原始CSV开始

```powershell
python scripts/run_baseline.py --mode preprocess-run --raw-dir "<本地21UBE0022日CSV目录>"
```

原始路径只在命令行提供。此模式不会自动重建电堆档位表；若该表缺失，需要另外提供刘占伟Stage 6成本表：

```powershell
python scripts/02_build_stack_degradation_h2.py --liu-cost-table "<本地成本表.csv>" --output data/processed/current_point_degradation_h2.csv
```

## 6. 输入输出与本地依赖

| 步骤 | 输入 | 输出 | 是否需私有原始数据 |
|---|---|---|---|
| 01预处理 | 7个21UBE0022日CSV | canonical、预处理摘要 | 是，仅重新生成时 |
| 02档位表 | 刘占伟Stage 6成本表 | current-point proxy/H2表 | 是，仅重新生成时 |
| 03动力学 | canonical | baseline_power_demand、动力学指标 | 否，需本地canonical大表 |
| 04预测 | baseline_power_demand | 预测轨迹、预测指标 | 否，需本地processed大表 |
| 05分配 | 需求、预测、档位、YAML | 四策略轨迹/指标/裁剪 | 否，需本地大表 |
| 06汇总 | 小型指标和轨迹 | summary、PNG/PDF | 轨迹为本地生成 |
| check | 已提交小型CSV/JSON | 通过/失败信息 | 否 |

## 7. 仓库维护规则

- baseline修改只进入`configs/baseline.yaml`、`scripts/run_baseline.py`和`scripts/baseline/`。
- 优化实验保留在`scripts/07--12`与`data/results/optimization/`，不得悄悄替换baseline结果。
- 大型generated CSV保留本地并由`.gitignore`排除；只提交代码、配置、小型指标、报告和图。
- 改动后先运行`--mode check`、Black、compileall，再提交。
