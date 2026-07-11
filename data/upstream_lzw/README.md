# 刘占伟退化代理上游关键数据

本目录保存当前 Gamma/健康模型和 degradation proxy 讨论所需的轻量上游文件，方便新环境不依赖完整 MATLAB 工程也能继续分析。

## 文件说明

| 文件 | 内容 | 用途 |
|---|---|---|
| `canonical_event_table_6104.csv` | 删除 `3700:3900` 后的 6104 个稳定事件表，已逐行对齐 θ | 暴露量、当前电流、累计持续时间、θ 对齐 |
| `theta_event_trajectory_6104.csv` | 带 `event_id/canonical_row/original_index` 的完整逐事件 θ 轨迹 | 推荐给后续健康模型直接使用 |
| `theta_ukfpf_physical.csv` | 从 UKF-PF 输出提取的原始物理 θ 三列 | 保留上游 θ 视图 |
| `theta_ukfpf_metadata.json` | θ 的 MAT 来源、列号、单位和 SHA-256 | 溯源说明 |
| `current_point_degradation_cost_table.csv` | early/middle/late × 典型电流点的完整性能损失表 | 构造/审计 `C_deg(I|θ)` |
| `health_state_theta_summary.csv` | early/middle/late 代表 θ 状态 | 健康阶段对照 |
| `current_point_cost_conditions.json` | IV_model 固定参考工况和参数 | 复算电流点 proxy |

## 关键边界

- `θ(k)=[i0(k), ih(k), R_ohm(k)]` 是 6104 个事件上的健康状态轨迹，不是按电流点分别辨识的一组参数。
- `current_point_degradation_cost_table.csv` 中的电流点差异来自同一个 IV_model 在不同参考电流下的模型评价。
- `R_model_state_for_iv_model = R_reported / 406` 是阶段 6 审计推断，不是作者 MATLAB 代码中明示的单位转换。
- 当前文件足以支持 degradation proxy / Gamma 健康骨架研究；若要完全复现观测修正或 UKF-PF 辨识，还需要完整 MATLAB 数据和观测输入说明。

