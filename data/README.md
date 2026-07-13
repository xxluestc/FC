# 数据使用说明

本目录只保存可复现处理链路的输出。原始整车 CSV、MATLAB MAT 文件和论文不提交 Git；大体积中间表默认由脚本在本地重建。当前实验没有把韩成杰或其他论文中的退化系数混入刘占伟数据链。

## 0. 新环境如何恢复关键数据

仓库提供了可接手运行的压缩关键数据：

```text
data/key/
```

下载仓库后先运行：

```bash
python scripts/00_materialize_key_data.py
```

这会恢复当前 baseline 需要的 processed CSV，包括：

- `processed/liu_vehicle_canonical_1s.csv`
- `processed/baseline_power_demand.csv`
- `processed/power_demand_from_dynamics.csv`
- `processed/baseline_prediction_results.csv`
- `processed/prediction_results.csv`
- `processed/current_point_degradation_h2.csv`

如果还要恢复李俊豪三台双堆发动机的三日 processed 表：

```bash
python scripts/00_materialize_key_data.py --include-li-junhao
```

注意：这些仍然是处理后的关键数据，不是原始 CSV/XLSX/MAT。原始数据、论文 PDF 和本地路径配置不进入 Git。

## 1. 当前使用的数据

整车需求与预测主链使用刘占伟目录中的 7 个 `21UBE0022` 日文件：

`论文代码/半个月的数据/原始数据/原始数据/原始数据/21UBE0022_苏E02625F-2025-05-*.csv`

- 原始记录：106,162 行；去除 278 个重复时间戳后为 105,884 个唯一记录。
- 覆盖日期：2025-05-17 08:10:14 至 2025-05-27 09:55:30。
- 日期跨度约 10.07 天，但不是连续十天采样；按大于 10 s 的间断划分为 46 个有效运行片段。
- 各片段内重采样为 1 s 后得到 211,630 行，其中禁止跨片段、跨停车或跨夜插值。
- 车辆标识来自文件名：`21UBE0022 / 苏E02625F`。

退化代理来自刘占伟另一条 MATLAB 链中的 6,104 个 `x_est=[i0, ih, R_ohm]` 事件状态及电压模型性能损失表。尚无车辆号或电堆编号证明该 `x_est` 与上述 `21UBE0022` 半月 CSV 为同车同堆，因此只能称为 **degradation proxy（退化代理）**，不能解释为逐行同源的真实退化系数。

### 独立跨月功率验证

`G:/大论文/实车数据/21UBE0022_苏E02625F`的2025-06至2026-06归档不与上述五月七天链拼接。
它当前有三个分开的用途：全年功率包络支持40 kW经验参考，匹配电压用于健康信号可辨识性审计，
以及从13个月各三个时间层抽取39个30分钟块作冻结控制器的外部功率回放。

跨月回放只读取时间戳、DCDC输入电压和输入电流。70,200个基础1秒点中35,117点是在不超过10秒
的连续遥测段内由相邻包插值得到；任何大于10秒的缺口均不跨越。每块重置健康和控制状态，因此
该数据验证的是不同月份真实功率条件下的可执行性和健康均衡方向，不是同一物理电堆连续13个月
的SOH或寿命验证。归档中的突降和时期台阶不直接写回动作驱动累计退化`D`。

## 2. canonical 字段

`processed/liu_vehicle_canonical_1s.csv` 的主要字段如下。

| 字段 | 单位 | 含义/来源 |
|---|---:|---|
| `timestamp` | 日期时间 | 原始上报时间，去重后按片段重采样 |
| `segment_id` | - | 相邻有效记录间隔大于 10 s 时新建片段 |
| `speed_kmh`, `speed_mps` | km/h, m/s | 原始整车车速及单位换算 |
| `acceleration_mps2` | m/s² | 片段内车速一阶差分，禁止跨段计算 |
| `soc_pct` | % | 动力电池 SOC |
| `target_power_kw` | kW | 车辆控制器原始目标功率字段 |
| `fc_voltage_v`, `fc_current_a` | V, A | 燃料电池系统测量量 |
| `dcdc_output_voltage_v`, `dcdc_output_current_a` | V, A | DCDC 输出测量量 |
| `battery_voltage_v`, `battery_current_a` | V, A | 动力电池测量量，保留原始符号 |
| `motor_voltage_v`, `motor_current_a` | V, A | 驱动电机测量量，保留原始符号 |
| `single_h2_kg`, `cumulative_h2_kg` | kg | 原始单次/累计耗氢字段，尚未直接用作 MPC 氢耗率 |
| `loadable_power_kw` | kW | 原始可加载功率字段 |
| `odometer_km` | km | 原始累计里程 |
| `mean/min/max_cell_voltage_v` | V | 85 节电池单体电压统计 |
| `fc_input_power_kw` 等 | kW | 由对应电压×电流/1000 派生，保留传感器符号 |

`processed/power_demand_from_dynamics.csv` 在 canonical 表上增加：平滑车速/加速度、实测需求功率 `p_dem_measured_kw`、车辆动力学基线和分牵引/制动校准功率。主预测目标是同表实测需求功率；动力学功率只作独立基线，二者不能混称。

预测器额外以因果方式构造三个停车状态特征：当前连续停车时长 `stop_duration`、距最近一次停车时间 `time_since_last_stop`、距最近一次停车后的累计里程 `distance_since_last_stop`。它们在运行时计算，不重复写入大表。

字段可用性审计进一步确认：当前表没有 DCDC 目标电流、明确 FC 状态、空气流量、温度、压力、空压机/水泵/氢泵/风扇/电加热器，也没有完整单体电压数组。上述变量不能在当前实验中假定存在；若后续需要，必须回到原始CSV扩充字段映射并重新生成canonical。逐字段结果见 `results/optimization/processed_feature_availability.csv`。

## 3. 处理流程

```text
7 个 21UBE0022 原始日 CSV
  -> 字段语义映射、时间解析、同时间戳聚合
  -> 按 >10 s 间断分成 46 个 segment
  -> segment 内 1 s 重采样和短缺口插值
  -> canonical 车速/SOC/FC/电池/电机表
  -> 实测需求功率 + 独立车辆动力学功率校准
  -> 历史 30 s 状态 + 停车状态 + 工况标志
  -> H=1/3/5/10 horizon-specific 功率预测
  -> 预测置信衰减后的滚动 MPC
```

对应脚本为 `scripts/01_preprocess_data.py`、`03_vehicle_dynamics_power.py`、`04_train_or_run_predictors.py` 和 `05_run_power_allocation.py`。

## 4. 目录内输出

- `processed/current_point_degradation_h2.csv`：刘占伟典型电流档位对应的电堆功率、退化代理和法拉第理论氢耗。
- `processed/prediction_results.csv`：各预测起点、预测域、步长、验证集选中模型、预测/真实功率。
- `results/prediction_metrics.csv`：模型族在验证集和测试集的逐秒、窗口能量和事件指标。
- `results/allocation/allocation_trajectory.csv`：各策略逐秒需求功率、FC/电池功率、SOC、档位及逐项成本。
- `results/allocation/mpc_weight_search.csv`：只在校准前缀上完成的权重与置信衰减搜索。
- `results/allocation/*diagnostics.csv`：档位占用、各档 proxy 贡献、制动分段以及 Predicted/Perfect 动作差异。
- `results/optimization/`：字段审计、字段组消融、控制结构/权重搜索、Pareto候选和SOC等值对照。
- `results/baseline/`：冻结XGBoost与固定权重四策略的小型指标、裁剪审计和汇总；大型预测/轨迹表仅本地生成。

百分比误差使用测试需求功率全量程归一化；报告同时保留 kW 原值，不能只引用较小的百分数。功率分配前对超出 FC+电池约束的需求进行裁剪，裁剪审计位于 `demand_clipping_audit.json`，因此控制结论属于可行域内实验。
