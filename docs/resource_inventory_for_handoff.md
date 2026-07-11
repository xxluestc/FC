# 当前资源总览与接手说明

本文档给后续接手者快速判断：本仓库已经包含什么、哪些外部资源曾被参考、哪些材料仍需要用户单独提供。当前主线仍是：

```text
刘占伟老化参数/IV模型 → 电流点 degradation proxy
李俊豪双堆实车数据可用性审计
短时需求预测 + 氢电功率分配 baseline
```

## 1. 仓库已内置的关键数据

为方便新环境继续运行，仓库已提供轻量压缩数据包：

```text
data/key/
```

包括：

| 文件 | 内容 | 用途 |
|---|---|---|
| `liu_vehicle_canonical_1s.csv.gz` | 刘占伟整车 processed/canonical 1s 表 | 需求功率、预测、baseline 控制 |
| `baseline_power_demand.csv.gz` | 已构建的需求功率表 | MPC/功率分配输入 |
| `baseline_prediction_results.csv.gz` | baseline 预测器输出 | Predicted MPC 对照 |
| `current_point_degradation_h2.csv.gz` | 电流点退化代理 + 理论氢耗表 | FC 档位成本 |
| `li_junhao_dual_stack/engine3_three_day_processed.csv.gz` | 李俊豪三号双堆三日 processed 表 | 双堆 A/B 电压与功率贡献审计 |
| `li_junhao_dual_stack/engine4_three_day_processed.csv.gz` | 李俊豪四号双堆三日 processed 表 | 同上 |
| `li_junhao_dual_stack/engine5_three_day_processed.csv.gz` | 李俊豪五号双堆三日 processed 表 | 同上 |

新环境中可运行：

```bash
python scripts/00_materialize_key_data.py
```

如需恢复李俊豪三日 processed CSV：

```bash
python scripts/00_materialize_key_data.py --include-li-junhao
```

这些是处理后的关键数据，不是原始大数据。原始 Excel/MAT/PDF 未提交。

## 2. 刘占伟资源

### 2.1 当前用途

刘占伟数据是当前燃料电池退化代理主线的数据来源。已经使用的内容包括：

```text
data_mark 稳定事件表
UKF-PF 老化参数 θ=[i0, ih, R_ohm]
IV_model 极化曲线模型
电流点性能损失表
```

当前表述必须是：

```text
degradation proxy / performance loss proxy
```

不能写成真实材料退化系数。

### 2.2 关键链路

```text
刘占伟原始/中间 MAT 数据
        ↓
data_mark 稳定事件表
        ↓
删除 MATLAB rows 3700:3900
        ↓
6104 个稳定事件
        ↓
对齐 UKF-PF 保存的 θ(k)
        ↓
IV_model 固定参考工况
        ↓
C_deg(I|θ)
```

### 2.3 仓库内相关文件

```text
data/processed/current_point_degradation_h2.csv
data/key/current_point_degradation_h2.csv.gz
docs/handoff_degradation_proxy_context.md
reports/degradation_h2_model.md
scripts/02_build_stack_degradation_h2.py
```

### 2.4 仓库外曾用/待查文件

这些在上级工作目录中存在过，但未复制进本仓库：

```text
../stage_outputs/stage_04/data/canonical_event_table_6104.csv
../stage_outputs/stage_04/reports/canonical_event_table_report.md
../stage_outputs/stage_06/reports/iv_model_translation_report.md
../stage_outputs/current_point_degcost/data/current_point_degradation_cost_table.csv
../stage_outputs/current_point_degcost/reports/current_point_degradation_cost_audit.md
../scripts/iv_model/iv_model.py
../scripts/lzw_pipeline/current_point_degcost/build_current_point_cost.py
```

接手者若需要复核“原始 MAT → data_mark → θ”的完整来源，应要求用户提供刘占伟 MATLAB 工程或读取上级目录。

## 3. 李俊豪资源

### 3.1 当前用途

李俊豪数据用于评估“双堆实车数据是否适合后续多堆退化与功率分配研究”。当前结论：

- 三号、四号、五号发动机均有 A/B 双堆字段；
- 数据中 A/B 共用同一燃料电池电流，更像串联系统；
- 可以分析 A/B 电压衰减、不一致性、弱堆识别、氢耗字段和 FC 工作点；
- 不能直接作为“两个电堆独立可控功率分配”的监督标签；
- 没有完整电池功率字段时，无法严格得到真实 `P_dem = P_fc + P_bat` 分解。

### 3.2 仓库内相关文件

```text
scripts/13_li_junhao_voltage_audit.py
scripts/14_inspect_li_junhao_daily_schema.py
scripts/15_process_li_junhao_dual_stack.py
scripts/16_finalize_li_junhao_dual_stack_audit.py
experiments/li_junhao_voltage_audit/
experiments/li_junhao_dual_stack_audit/
data/key/li_junhao_dual_stack/
```

### 3.3 仓库外原始资源

用户本地路径曾为：

```text
H:/其他/2026李俊豪/学位论文/实验数据
H:/其他/2026李俊豪/学位论文/实验模型/process_data
H:/其他/2026李俊豪/学位论文/实验模型/process_data/datasets
```

原始 Excel 很大且包含多个 sheet，未入库。仓库中只保存了字段审计、小表、图和三台发动机三日 processed 压缩表。

## 4. 韩成杰资源

### 4.1 当前用途

韩成杰项目主要作为功率分配/能量管理框架参考，尤其是：

- 氢电功率分配结构；
- 锂电池/SOC模型；
- 需求功率测试输入与工况划分思路；
- 优化目标中氢耗、电池、SOC、平滑项的组织方式。

重要边界：

```text
韩成杰燃料电池退化系数不是刘占伟实车数据辨识得到的，
当前项目不能直接使用它作为燃料电池退化系数。
```

当前燃料电池退化 proxy 必须来自刘占伟链路。

### 4.2 仓库状态

韩成杰论文/代码未纳入本仓库。若接手者需要复核锂电池模型或原始工况划分，请让用户单独提供韩成杰目录或相关论文/代码。

用户曾提到的参考目录：

```text
H:/毕业生文件-韩成杰--工况
```

当前仓库只保留了从该思路抽象出来的 baseline 控制框架，不保留原始论文/私有数据。

## 5. 陈鹏资源

当前对话中用户提到“陈鹏等人的一些关键代码或者论文内容”，但本仓库尚未纳入陈鹏相关代码、论文或数据，也未完成目录审计。

接手者应按以下方式处理：

1. 不要假设陈鹏资源内容；
2. 若需要引用或复现，应让用户单独提供目录或文件；
3. 收到后先做文件清单、论文摘要、代码用途说明，再判断能否接入当前主线。

建议陈鹏资源若后续接入，优先整理为：

```text
docs/external_resources/chen_peng_notes.md
experiments/chen_peng_audit/
```

## 6. 当前主线代码结构

```text
scripts/00_materialize_key_data.py       # 解压关键数据
scripts/baseline/                        # 最小 baseline pipeline
scripts/02_build_stack_degradation_h2.py # 刘占伟退化代理+氢耗表
scripts/05_run_power_allocation.py       # 功率分配实验
scripts/10_optimize_predictive_controller.py # 优化实验，不是 baseline 默认
scripts/13-16_li_junhao_*.py             # 李俊豪数据审计
src/fc_power/                            # 模型与控制器模块
data/key/                                # 可接手运行的压缩关键数据
data/results/                            # 轻量结果表
docs/                                   # 流程、交接和资源说明
reports/                                # 实验报告
```

## 7. 当前最重要的研究边界

1. 当前 `C_deg(I|θ)` 是退化代理，不是真实材料退化率。
2. 当前功率分配里 θ 尚未随新决策在线演化。
3. 已实现的是“给定健康状态下，不同电流点的性能损失代理”。
4. 若要实现闭环退化状态更新，需要进一步建模：

```text
θ(t+1) = θ(t) + f(action, exposure, temperature, pressure, start-stop, ramp)
```

5. 刘占伟数据可支持 θ 轨迹和 IV proxy；李俊豪数据可支持双堆 A/B 电压衰减审计；韩成杰可参考 EMS/电池/控制结构；陈鹏资源待用户补充。

## 8. 新接手建议

第一步：

```bash
python scripts/00_materialize_key_data.py
python -m compileall scripts src
```

第二步先读：

```text
README.md
docs/baseline_project_overview.md
docs/baseline_runbook.md
docs/handoff_degradation_proxy_context.md
docs/resource_inventory_for_handoff.md
reports/baseline_pipeline_report.md
```

第三步如果要继续当前问题，优先围绕：

```text
如何把固定 late 表升级为 C_deg(I|θ_current)
如何在功率分配中解释 θ 不随决策在线更新这一边界
是否需要重新做暴露量 → Δθ 的轻量模型
```

