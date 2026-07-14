# Codex CLI交接：物理边界与Rate-bounded Blend

更新时间：2026-07-14

> 后续已完成简化退化随机负载全链路，最新交接见
> `docs/CODEX_CLI_HANDOFF_2026-07-14_SIMPLIFIED_DEGRADATION_CHAIN.md`。

## 1. Git与验收

- 仓库：`H:/其他/2026刘展玮/FC`
- 分支：`codex/gamma-health-foundation`
- 本轮研究代码与结果提交：`fe55b39`（`research: bound Blend by physical rate domain`）
- 上一交接提交：`2deb2f0`
- 远端/本地`main`基准：`7e71f396107751d9d0e93a19da60249e8c8a9122`
- 120项单元测试全部通过；baseline轻量检查、`compileall`和`git diff --check`通过。

## 2. 本轮解决了什么

此前Guarded Blend只判断“当前最老堆是否也是估计退化最快堆”。本轮先把健康停止边界从
LZW标定终点扩展到定工况5%/10%电压损失压力场景，再固定退化过程只移动停止边界。结果发现：

- LZW终点`D=9.353029%`不是物理EOL；它在195 A和370 A下仅对应约1.71%和3.24%电压损失；
- 5%、10%、15%电压损失边界全部超出LZW实测参数轨迹，必须标成外推压力场景；
- 强同向异质性下，原Guarded Blend在LZW终点的N+1第二边界平均下降16.45小时，95%区间
  `[-41.90,-1.40]`，说明身份排序正确仍不代表Blend安全；
- 0.25、0.50、0.75、0.90、0.99五个既有权重均未通过强同向场景的N+1下界非负门槛，
  简单调权被否决。

因此慢层候选升级为`Rate-bounded Blend`：

1. 三堆当前健康必须有区分；
2. 最老堆与估计退化最快堆必须是同一身份；
3. `max(relative_rate)/min(relative_rate) <= 1.10`；
4. 三项同时满足才执行Blend 0.50，否则回退`fixed_pair`。

1.10来自项目既有名义异质性因子`(1.00,1.05,1.10)`，不是按本轮结果反推。该规则已实现为
`select_guarded_blend_policy`并有单元测试。

## 3. 核心结果

| 审计 | 原Guarded Blend | Rate-bounded Blend |
|---|---:|---:|
| 41参数场景启用Blend | 18 | 12 |
| 41参数场景回退fixed | 23 | 29 |
| 41场景N+1负CI | 0 | 0 |
| 跨月参考首/第二边界 | +71.07/+4.08 h | +71.07/+4.08 h |
| 跨月强同向首/第二边界 | +50.77/+1.16 h，第二CI跨0 | 0/0 h |
| 跨月强错配首/第二边界 | 0/0 h | 0/0 h |

新门控的代价是放弃强同向异质性下约50.77小时首边界收益；收益是不会在未标定的大速率离散
条件下继续冒险。它是带明确适用域的保守方法，不是任意异质性下普适最优。

物理边界实验为4边界×3异质性×20配对种子，仿真上限提升到10000小时后全部到达第二边界。
初始绝对损伤、Gamma尺度、退化系数、实车模板和策略参数均冻结；边界变化没有与退化过程缩放
混杂。逐种子跨边界差分也已输出。

## 4. 代码、结果与图

- `scripts/64_audit_voltage_loss_boundary_mapping.py`：LZW损伤到电压损失边界映射；
- `scripts/65_audit_frozen_process_physical_boundaries.py`：冻结过程的物理边界压力测试；
- `scripts/66_screen_blend_weight_strong_heterogeneity.py`：强同向异质性权重筛查；
- `scripts/67_audit_rate_bounded_blend.py`：41场景、物理边界、跨月三层重放；
- `data/results/fc_only_physical_boundary_mapping/`：Fig.27与边界表；
- `data/results/fc_only_frozen_process_physical_boundaries/`：Fig.28与20种子结果；
- `data/results/fc_only_blend_weight_strong_heterogeneity/`：Fig.29与权重否证；
- `data/results/fc_only_rate_bounded_blend/`：Fig.30与三层审计；
- `data/results/figures/fc_only_foundation/fig27...fig30...png`：320 DPI PNG汇总。

长代表轨迹保留本地并由`.gitignore`忽略；仓库提交逐运行指标、配对差分、汇总、元数据和图，
可以由脚本完整再生。

## 5. 复现命令

```powershell
cd 'H:\其他\2026刘展玮\FC'
$env:PYTHONPATH='src'

python scripts\64_audit_voltage_loss_boundary_mapping.py
python scripts\65_audit_frozen_process_physical_boundaries.py --seeds 20 --jobs 16 --max-hours 10000
python scripts\66_screen_blend_weight_strong_heterogeneity.py --seeds 20 --jobs 16 --max-hours 6000
python scripts\67_audit_rate_bounded_blend.py

python -m unittest discover -s tests -v
python scripts\run_baseline.py --mode check
python -m compileall -q src scripts tests
git diff --check
```

## 6. 数据边界

`G:/大论文/实车数据/21UBE0022_苏E02625F`继续只用于独立功率负载块和聚合性能信号审计。
它只有整车/单聚合堆通道，不能辨识同一三堆系统中三只堆的相对退化速率，也没有给本轮1.10
门槛提供车辆拟合证据。突然电压衰减和时期台阶未写回健康`D`。

所有5%/10%电压损失边界都是LZW模型外推，不称真实EOL，不把边界到达小时数称真实RUL。

## 7. 下一顺序与需要的外部资料

基础主线继续按以下顺序：

1. 建立三只真实堆相对退化速率的可部署观测/检测口径，并给出排序与速率比不确定性；
2. 获得真实EOL或维护判据后替换外推停止边界；
3. 获得启停、换堆和非计划停机成本后，把当前`h/start`边界收益换算为维护代价；
4. 在上述口径稳定后再考虑机理残差世界模型和Dreamer/EAWM；TD-MPC仍只适合后续连续动作方向；
5. 锂电池外层和未来需求预测继续冻结。

若用户能向老师或师兄补资料，最有价值的是：同一三堆系统中每只堆独立的电压、电流、温度、
气体压力/流量、稳定`stack_id`、维护/换堆记录、厂家EOL判据，以及一次启停/换堆/停运的实际
成本。这些资料分别用于辨识相对速率、验证停止边界和确定门控保守程度；缺失时不阻塞代码研究，
但不能把离线压力门控称为真实车辆在线部署结果。

## 8. 代理复核

本地Claude Code实际接DeepSeek。本轮让其只读检查冻结过程边界设计；它正确提醒了外推误差和
20种子不确定性，但关于“固定绝对初始损伤会使Guard跨边界改变来源”的判断不成立，因为统一
正比例缩放不改变健康排序或是否分离。代理没有修改仓库，所有结论均由代码和配对结果复核。
