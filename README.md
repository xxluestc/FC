# FC：退化代理感知的燃料电池功率调控

本仓库维护以下可审计最小路线：

```text
刘占伟21UBE0022整车原始数据
  → 时间轴/车速/功率/SOC预处理
  → 车速预测与需求功率预测
  → 韩成杰车辆动力学换算与实测功率校准
  → 刘占伟老化参数性能损失表形成 degradation proxy
  → 氢耗、电池和SOC模型
  → Instant / Constant / Perfect / Predicted MPC对比
```

## 数据边界

- 原始数据、MAT、论文和本地绝对路径不进入Git。
- 需求功率主链来自刘占伟21UBE0022半月CSV，同表包含时间、车速、SOC、FC、电池和电机信号。
- `degradation proxy` 来自刘占伟6104事件老化参数/性能损失表，但尚未证明该 `x_est` 与21UBE0022半月CSV为同车同堆，因此不能称为真实退化系数，也不能逐行对齐解释。
- 韩成杰数据用于车辆动力学参数、运动学片段和Markov方法参考；李俊豪数据用于多堆和参数辨识参考。

## 代码框架

| 层 | 文件 | 职责 |
|---|---|---|
| 数据 | `src/fc_power/data/` | 数据源审计、字段映射、去重、分段和重采样 |
| 动力学 | `vehicle_dynamics.py` | 空阻、滚阻、坡阻、加速阻力和轮端功率 |
| 电堆 | `stack_model.py` | 电流—电压—功率映射 |
| 氢耗 | `hydrogen_model.py` | 法拉第理论氢耗 |
| 退化代理 | `degradation_cost.py` | 刘占伟性能损失表插值；不表示真实材料退化率 |
| 电池 | `battery_model.py` | Rint电流、SOC更新和吞吐量 |
| 预测 | `prediction/` | 速度预测、直接功率预测、物理残差校正 |
| 分配 | `power_allocation/` | 瞬时与滚动优化、SOC和功率约束 |
| 评价 | `evaluation/` | MAE/RMSE、百分比指标和科研图 |

预测方法严格区分：

1. `speed_only_dynamics`：预测速度，再由动力学换算功率；
2. `state_direct_power`：历史速度、加速度、功率和工况状态直接预测功率；
3. `hybrid_physics_corrected`：速度动力学基线加数据驱动残差。

## 脚本顺序

| 脚本 | 输出 |
|---|---|
| `00_audit_data.py` | 数据字段与来源清单 |
| `01_preprocess_data.py` | 1秒canonical整车表 |
| `02_build_stack_degradation_h2.py` | 电流档位、degradation proxy和理论氢耗 |
| `03_vehicle_dynamics_power.py` | 动力学功率与校准指标 |
| `04_train_or_run_predictors.py` | 三类预测结果和H=1/3/5/10/15指标 |
| `05_run_power_allocation.py` | 四策略、三时域、敏感性和裁剪审计 |
| `06_run_all_experiments.py` | 已生成结果的文件、schema和SOC约束检查 |
| `plot_results.py` | 论文风格PNG/PDF图 |

## 当前验证结论

- 车辆动力学测试：MAE 8.55 kW、RMSE 17.85 kW、R² 0.877。
- `state_direct_power` 的窗口平均功率在H=1/3/5时，NMAE与NRMSE均低于5%。
- H=10/15的NMAE低于5%，但NRMSE约5.76%/5.78%，未达到5%硬门槛；默认控制时域因此设为H=5。
- Predicted MPC相对Constant MPC有改善，但相对Instant并非所有指标占优。
- 测试需求存在1.75%点被裁剪，故功率分配结论限定为可行域内实验。

百分比采用测试集功率全量程归一化：

`NMAE = MAE / (Pmax-Pmin) × 100%`，`NRMSE = RMSE / (Pmax-Pmin) × 100%`。

报告同时保留 `RMSE / actual RMS`，防止只挑有利分母。

## 本地执行

复制 `configs/paths.template.yaml` 为不提交的 `configs/paths.local.yaml`，然后按顺序运行编号脚本。主要命令示例：

```powershell
python -m black --check src scripts
python -m compileall src scripts
python scripts/06_run_all_experiments.py
python scripts/06_run_all_experiments.py --full-local
python scripts/plot_results.py
```

完整参数和输入路径见各脚本 `--help`。阶段结论位于 `reports/`，架构与数据来源位于 `docs/`。
