# 交接文档：退化代理代价来源审查与当前研究上下文

本文档用于新对话/新 AI 快速接手当前项目。当前用户最关心的问题不是继续优化控制器，而是彻底说清楚：

```text
当前用于功率分配的“不同电流点退化代价”到底从哪里来？
它和刘占伟原始数据、老化参数、IV模型、电压之间是什么关系？
它能否作为后续氢电/多堆功率分配的决策代理？
```

## 1. 当前项目目标

用户的长期目标是做：

```text
多堆燃料电池 / 锂电池功率调控中的退化感知功率分配
```

当前阶段不是追求真实材料退化系数的严格因果辨识，而是先构造一个可以接入 EMS/MPC/功率分配目标函数的数值型退化代理：

```text
C_deg(action | state)
```

允许它是 proxy，但必须来源清楚、逻辑自洽、不能混用韩成杰或其他论文的燃料电池退化系数。

当前已经跑通的最小路线是：

```text
刘占伟 UKF-PF 老化参数 θ
        ↓
刘占伟 IV_model
        ↓
不同电流点下的等效性能损失 D(I,θ)
        ↓
degradation proxy 档位表
        ↓
氢电功率分配目标函数
```

## 2. 最重要结论

当前不同电流点退化代价不是“实测得到的电流点真实退化系数”。

它是：

```text
基于刘占伟 UKF-PF 老化参数 θ 和 IV_model 构造的
不同电流工作点下等效性能损失代理
```

推荐表述：

```text
degradation proxy / performance loss proxy
```

不要表述为：

```text
真实退化系数
材料退化速率
某电流动作导致的未来退化增量
```

用户已经明确：知道它不能当真实退化系数，但可以作为决策代理使用。

## 3. 关键回答：老化参数是否分电流点？

不分。

刘占伟原始老化参数是一条随事件序号变化的健康状态轨迹：

```text
θ(k) = [i0(k), ih(k), R_ohm(k)]
k = 1,2,...,6104
```

每个 `k` 对应一个稳定事件/工况片段，不是某个电流点专属的参数。

错误理解：

```text
θ_25A, θ_90A, θ_120A, θ_195A, ...
```

正确理解：

```text
同一个 θ(k) 表示当前健康状态；
后处理时把这个 θ(k) 代入不同参考电流 I，
得到不同 I 下的模型电压损失。
```

因此不同电流点的 proxy 不是作者原始辨识时分组得到的，而是通过 IV_model 后处理得到的“反事实工作点评估”。

## 4. IV_model 是什么？

不是“每个电流点一个模型”，而是一个统一的燃料电池极化曲线模型：

```text
V = E - η_act - η_ohm - η_con
```

其中：

- `E`：可逆电压，受温度、氢/氧条件影响；
- `η_act`：活化损失，主要受 `i0`、`ih`、电流密度影响；
- `η_ohm`：欧姆损失，主要受 `R_ohm`、电流密度影响；
- `η_con`：浓差损失，主要受极限电流、气体条件、电流密度影响。

当前 Python 翻译文件：

```text
../scripts/iv_model/iv_model.py
```

上游 MATLAB 模型来源于刘占伟工程中的 `IV_model.m` / `UKF_PF.m` 内嵌模型。Python 版本已经做过 MATLAB oracle 级函数一致性验证。

相关报告：

```text
../stage_outputs/stage_06/reports/iv_model_translation_report.md
```

## 5. 为什么用 θ 代入不同电流得到退化 proxy？

同一个健康状态在不同电流下表现出的性能损失不同。直观例子：

```text
同样 R_ohm 增大，
低电流时电压损失小，
高电流时电压损失大。
```

所以当前 proxy 计算的是：

```text
当前健康状态 θ 下，
如果电堆运行在 I=90A/195A/270A/370A/...，
相对健康基线会多损失多少电压/等效功率。
```

它回答：

```text
当前老化状态下，不同工作电流点的性能损失敏感性。
```

它不回答：

```text
长期运行在某电流点会导致多少未来材料退化。
```

如果后续要做真实“工况暴露量 → Δθ”，需要另外建立模型；之前尝试过但效果不够稳定，所以当前路线退回到“性能损失 proxy 作为决策依据”。

## 6. 为什么不直接从电压得到退化？

直接原始电压不能简单等价为退化，因为实车/动态数据中的电压同时受很多因素影响：

```text
V = f(电流, 温度, 氢压, 空气压力, 氧浓度, 水热状态, 控制策略, 老化状态, 噪声)
```

例如高电流下电压低，不一定是退化严重；也可能只是负载高、温度低、供气不足或压力波动。

所以当前路线是：

```text
原始电压/压力/温度/电流
        ↓ UKF-PF
老化参数 θ
        ↓ IV_model 固定参考工况
不同电流点下的等效性能损失 D(I,θ)
```

这样可以在模型空间里固定温度、压力和气体条件，只改变电流，得到标准化比较。

## 7. “相同温度、压力、气体条件”是怎么做到的？

不是从原始数据中找完全相同工况样本，而是在 IV_model 中固定参考条件。

当前固定条件来自刘占伟 6104 事件链的中位/参考值：

```json
{
  "T_ref_C": 58.88708333333334,
  "a_ref": 1.1026842574415214,
  "b_ref": 0.19478201132151415,
  "B": 0.012356427043707334,
  "i_lim_A_cm2": 1.2015784940009275,
  "active_area_cm2": 406.0,
  "stack_cells": 170,
  "energy_reference_step_s": 1.0
}
```

对应文件：

```text
../stage_outputs/current_point_degcost/tables/current_point_cost_conditions.json
```

在这些条件固定后，只改变：

```text
I = 0,25,90,120,160,195,270,370 A
```

从而得到不同电流点下的 proxy。

## 8. 当前利用的原始数据到底是什么？

当前退化链使用的是刘占伟 MATLAB 工程，不是韩成杰，不是李俊豪，不是 21UBE0022 半月整车 CSV。

### 8.1 原始/中间运行数据

审计到的主要数据：

```text
论文代码\21022电堆单电池数据\工况信息与老化参数关系探究\data.mat
```

变量：

```text
data: 10927080 × 12
```

此外，稳定工况事件表来源：

```text
论文代码\电堆数据\工况数据提取\data_mark.mat
```

原始：

```text
data_mark: 6305 × 27
```

作者在多处 MATLAB 脚本中统一删除：

```text
MATLAB 1-based rows 3700:3900
```

删除 201 行后得到：

```text
6104 × 27
```

### 8.2 老化参数 θ 来源

首选 canonical 来源：

```text
论文代码\电堆数据\老化参数拟合\滤波算法\UKF-PF老化参数区间比较\all_data_UKF_PF.mat
```

其中 `data_UKF_PF` 的 MATLAB 第 2、6、10 列是点估计：

```text
i0, ih, R_ohm
```

便利副本：

```text
论文代码\21022电堆单电池数据\工况信息与老化参数关系探究\x_est.mat
```

形状：

```text
x_est: 3 × 6104
```

单位/缩放：

- `i0`：物理解释为 `stored × 1e-7 A/cm²`；
- `ih`：`A/cm²`；
- `R_ohm`：报告/表中约 `0.0406 → 0.0634`，进入 IV_model 时使用 `R_reported / 406`，这是审计推断，不是 MATLAB 中明示转换。

## 9. canonical 事件表

当前整理后的核心表：

```text
../stage_outputs/stage_04/data/canonical_event_table_6104.csv
```

它由以下内容逐行对齐得到：

```text
data_mark 删除 3700:3900 后的 6104 行
+
UKF-PF 的 6104 个 θ
```

主要字段包括：

```text
event_id
canonical_row_6104
original_index
current_A
previous_current_A
next_current_A
voltage_V
temperature_C
h2_pressure
air_pressure
I_step_A
abs_I_step_A
qt_num_cum
bz_num_cum
time_0A_cum
time_25A_cum
time_90A_cum
time_120A_cum
time_160A_cum
time_195A_cum
time_270A_cum
time_370A_cum
i0
ih
R_ohm
model_input_...
```

报告：

```text
../stage_outputs/stage_04/reports/canonical_event_table_report.md
```

关键结论：

- 6104 行与 UKF-PF `x_est=[i0,ih,R_ohm]` shape、顺序和数值逐元素一致；
- 删除 3700:3900 是作者人工规则，下游脚本普遍沿用；
- 但删除原因没有代码注释，不能断言是传感器异常、工况异常或 UKF 异常。

## 10. 从原始实车数据到老化参数的链路是否完全走通？

需要谨慎表述。

已经走通的是：

```text
刘占伟 MAT 数据审计
→ data_mark 稳定事件表
→ 删除 3700:3900
→ 6104 事件 canonical 表
→ 对齐作者保存的 x_est / all_data_UKF_PF
→ 得到 θ(k)
→ IV_model 映射到电流点 degradation proxy
```

还没有完全独立复现的是：

```text
从最原始 data.mat 开始，
完全用 Python 重新运行 UKF-PF，
逐点重新辨识出和作者 x_est 完全一致的 θ。
```

所以论文/报告中推荐写：

```text
老化状态参数采用刘占伟 UKF-PF 辨识结果；
本文进一步基于其 IV_model 构造不同电流工作点下的等效性能损失代理。
```

不要写：

```text
本文从原始实车数据完全独立辨识了所有老化参数。
```

除非后续真正复现 UKF-PF 全流程。

## 11. 当前电流点代价表是如何生成的？

上游脚本：

```text
../scripts/lzw_pipeline/current_point_degcost/build_current_point_cost.py
```

核心公式：

```text
D(I,θ)=V_model(I,θ0)-V_model(I,θ)
```

其中：

- `θ0`：early 健康基线，前 50 个事件均值；
- `θ_middle`：中间 50 个事件均值；
- `θ_late`：最后 50 个事件均值；
- `I`：`0/25/90/120/160/195/270/370 A`；
- `N_cell=170`；
- `active_area=406 cm²`。

健康状态摘要：

```text
../stage_outputs/current_point_degcost/tables/health_state_theta_summary.csv
```

典型值：

| 状态 | 事件范围 | i0_model_A_cm2 | ih_model_A_cm2 | R_reported_equivalent |
|---|---:|---:|---:|---:|
| early | 1–50 | 1.907e-7 | 0.002000 | 0.04060 |
| middle | 3028–3077 | 1.878e-7 | 0.003182 | 0.04919 |
| late | 6055–6104 | 1.804e-7 | 0.003660 | 0.06342 |

完整上游表：

```text
../stage_outputs/current_point_degcost/data/current_point_degradation_cost_table.csv
```

审计报告：

```text
../stage_outputs/current_point_degcost/reports/current_point_degradation_cost_audit.md
```

## 12. 当前 FC 仓库实际使用的代价表

FC 仓库中使用的是二次整理后的档位表：

```text
data\processed\current_point_degradation_h2.csv
```

生成脚本：

```text
scripts\02_build_stack_degradation_h2.py
```

它做了这些事：

1. 读取上游 current-point cost table；
2. 默认筛选 `health_state == late`；
3. 计算电堆功率：

```text
stack_power_kw = current_A × V_aged_cell_V × 170 / 1000
```

4. 保留：

```text
performance_loss_cost_raw_wh_step
performance_loss_cost_clipped_wh_step
performance_loss_cost_normalized
```

5. 另外计算法拉第理论氢耗：

```text
faraday_h2_g_s
```

当前最终表：

| current_a | stack_power_kw | aged_cell_voltage_v | raw Wh/1s | normalized |
|---:|---:|---:|---:|---:|
| 0 | 0.00 | 0.9061 | 0.0000 | 0.000 |
| 25 | 3.48 | 0.8191 | 0.0044 | 0.011 |
| 90 | 11.81 | 0.7716 | 0.0292 | 0.074 |
| 120 | 15.46 | 0.7579 | 0.0482 | 0.123 |
| 160 | 20.18 | 0.7420 | 0.0810 | 0.206 |
| 195 | 24.18 | 0.7296 | 0.1167 | 0.297 |
| 270 | 32.36 | 0.7049 | 0.2151 | 0.547 |
| 370 | 42.31 | 0.6727 | 0.3932 | 1.000 |

报告：

```text
reports\degradation_h2_model.md
```

## 13. 功率分配中如何调用

当前 FC 仓库配置：

```text
configs\baseline.yaml
```

关键配置：

```yaml
data:
  stack_map: data/processed/current_point_degradation_h2.csv

allocation:
  weights:
    degradation_proxy: 2.50

interpretation:
  degradation_proxy: "刘占伟老化参数/性能损失表形成的代理代价，不是真实材料退化系数。"
```

MPC 调用位置：

```text
src\fc_power\power_allocation\mpc_allocator.py
```

目标函数中相关项：

```python
weights["degradation_proxy"] * deg[nt]
```

其中 `deg[nt]` 就是当前候选燃料电池档位对应的 `performance_loss_cost_normalized`。

## 14. 对用户问题的标准回答模板

### Q1：现在不同电流点都有一个 IV 模型吗？

答：

不是。只有一个统一的 IV_model。不同电流点只是这个模型的不同输入。模型输入包括温度、气体条件、健康参数 θ 和电流密度；输出是该工作点下的单体电压。

### Q2：为什么一个 θ 可以代入不同电流？

答：

θ 表示当前健康状态，不是电流点参数。同一个健康状态在不同电流下表现出的电压损失不同。高电流会放大欧姆损失、活化损失和浓差损失，所以可以用 IV_model 评估不同工作电流下的性能损失。这个结果是“工作点性能损失 proxy”，不是“该电流造成的未来真实退化量”。

### Q3：这样能说在这个电流下的老化情况吗？

答：

不能严格说“这个电流造成的老化情况”。更准确说：

```text
在当前老化状态 θ 下，电堆运行在该电流时表现出的等效性能损失。
```

对于 EMS 决策，这个 proxy 可以用来偏好低性能损失的工作点。

### Q4：θ 不是主要和时间相关吗？怎么和工况相关？

答：

当前 θ 轨迹确实主要是随事件/时间变化的健康状态，没有被严格分解为各工况造成的因果贡献。工况通过 IV_model 的输入电流影响电压损失评价，而不是通过“电流导致 θ 变化”的方式进入当前 proxy。也就是说：

```text
θ 描述健康状态；
I 描述候选动作/工况；
IV_model 描述健康状态和动作共同决定的性能损失。
```

### Q5：相同温度压力条件怎么保证？

答：

不是在原始数据里筛同温同压样本，而是在模型里固定参考条件，然后只改变电流。这是模型标准化比较。

### Q6：是否完全走通原始数据到 θ？

答：

已走通“作者 MAT 结果链审计和对齐”，但没有完全用 Python 从最原始数据重新运行 UKF-PF 得到 θ。当前 θ 采用刘占伟 MATLAB 已保存的 UKF-PF 辨识结果。

## 15. 后续最合理的推进方向

如果用户继续追问或要改进，建议按这个顺序：

1. 不要再试图把当前 proxy 说成真实因果退化系数。
2. 把当前固定 `late` 表升级为健康状态相关版本：

```text
C_deg(I | θ_current)
```

而不是只用一张 late 静态表。

3. 做 early/middle/late 三套功率分配敏感性实验，说明健康恶化后 EMS 会更倾向避开高损失档位。
4. 如果要真正得到“工况暴露 → Δθ”，需要重新建模：

```text
z(k) 或窗口暴露量 → Δθ(k)
```

但之前 baseline 效果不好，需谨慎作为论文主线。
5. 如果用户要求“原始数据 → θ”全复现，则下一任务应是复现 `UKF_test_all.m` / `UKF_PF.m`，明确 `data_mark_update`、`a/b/bestX`、`V/v_test`、`R` 缩放，并用作者 `x_est` 做逐点误差验证。

## 16. 相关文件索引

### 刘占伟退化链

```text
../stage_outputs/stage_01/tables/mat_variable_summary.csv
../stage_outputs/stage_04/data/canonical_event_table_6104.csv
../stage_outputs/stage_04/reports/canonical_event_table_report.md
../stage_outputs/stage_06/reports/iv_model_translation_report.md
../stage_outputs/current_point_degcost/data/current_point_degradation_cost_table.csv
../stage_outputs/current_point_degcost/reports/current_point_degradation_cost_audit.md
```

### 关键脚本

```text
../scripts/lzw_pipeline/stage_04/build_canonical_event_table.py
../scripts/lzw_pipeline/stage_06/build_degradation_cost.py
../scripts/lzw_pipeline/current_point_degcost/build_current_point_cost.py
../scripts/iv_model/iv_model.py
```

### FC 仓库

```text
data\processed\current_point_degradation_h2.csv
scripts\02_build_stack_degradation_h2.py
reports\degradation_h2_model.md
configs\baseline.yaml
src\fc_power\power_allocation\mpc_allocator.py
```

## 17. 当前边界与风险

必须持续提醒：

- `x_est` 与 `data_mark` 对齐已经审计通过；
- 但未完全 Python 复现 UKF-PF 参数辨识过程；
- `R_reported/406` 是中等置信度单位适配；
- 当前 proxy 是性能损失代理，不是材料退化增量；
- 当前电流点差异来自 IV_model 的工作点评价，不是原始老化参数按电流点分组辨识；
- 当前 FC 功率分配中用的是 `late` 状态静态表，后续可以升级为 `C_deg(I|θ_current)`；
- 不要混用韩成杰燃料电池退化系数。韩成杰可以参考 EMS/锂电池建模/控制结构，但燃料电池退化 proxy 当前必须坚持来自刘占伟。

## 18. 给下一个 AI 的建议

用户现在的关注点很细：他会追问“这到底有没有物理意义”“数据到底从哪里来”“是不是偷换概念”。回答时请不要泛泛说“可以作为退化成本”，而要明确区分：

```text
健康状态 θ
候选动作 I
模型电压 V_model
等效性能损失 D(I,θ)
EMS proxy C_deg
```

最稳的表达：

```text
当前方案不是从每个电流点直接辨识真实退化速率；
而是利用刘占伟 UKF-PF 得到的健康状态参数，
在统一 IV 模型和固定参考工况下，
评价不同候选电流动作对应的等效性能损失。
这个量可以作为决策代理，但不能解释为动作导致的材料退化因果系数。
```

