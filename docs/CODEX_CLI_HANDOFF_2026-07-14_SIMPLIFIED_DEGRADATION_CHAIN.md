# Codex CLI交接：简化退化随机负载闭环

更新时间：2026-07-14

## 1. Git与验收

- 仓库：`H:/其他/2026刘展玮/FC`
- 分支：`codex/gamma-health-foundation`
- 本轮研究代码与结果检查点：`68536f886d29fd3e4829014cbabb4a8d9abcec47`
- 上一交接提交：`0af6fd8`
- 远端/本地`main`基准：`7e71f396107751d9d0e93a19da60249e8c8a9122`
- 127项单元测试全部通过；baseline轻量检查、`compileall`和`git diff --check`通过。

本文件在研究代码检查点之后单独提交，最终交接提交以`git log -1 --oneline`为准。

## 2. 本轮目标和结论

用户要求先把退化考虑进去并跑通整个路线，能看到Average、链式分配等策略的比较，以及Instant
考虑退化前后的对比；暂时不做效果优化。

这个目标已经完成。正式实验包含3类随机动态负载、5种策略、10个同种子，共150次180秒运行。
每一步都根据实际执行动作更新三堆健康，再通过`D -> theta -> IV/功率`进入下一步。150/150运行
步数完整、恰好两堆在线、每步健康更新；功率跟踪失败、约束违规、动作裁剪和安全接管全部为0。

因此当前可以确认的是“闭环已经跑通”，还不能确认“方法效果已经优化好”。

## 3. 当前实现

新增严格的Zuo式DC-average对照：按`(0,1) -> (1,2) -> (2,0)`顺序轮换在线堆，组内两堆
等电流分配。Average、Rotating、Instant-no-health和Instant-health与它面对相同负载、初始健康
和退化过程。

`Instant-no-health`只把规划目标中的退化增量和性能损失权重置零。被控对象仍按实际动作老化，
退化后的IV能力仍用于功率可行性。这避免了用“不会老化的对象”对比“会老化的对象”造成不公平。

退化暂时采用现有确定性动作驱动工程代理，保留连续运行、启停和变载损伤。短时Gamma随机采样、
锂电池、未来需求预测和实车健康观测校正均未进入本轮。

## 4. 正式结果

Instant-health相对Instant-no-health：

| 负载 | 最大单堆损伤增量 | 单位电量氢耗 | 跟踪MAE绝对变化 |
|---|---:|---:|---:|
| 实车1秒经验Markov | -0.91% | -0.10% | +0.023 kW |
| Zuo慢变30秒 | -2.21% | -0.10% | +0.022 kW |
| Zuo快变30秒 | -2.22% | -0.12% | +0.016 kW |

最大单堆损伤在10种子上分别为2胜8平、5胜5平和4胜6平，没有更差种子。实车经验负载的配对
区间跨0；两个Zuo压力场景的最大单堆损伤区间刚好低于0。多数种子仍选到相同结果，所以只能说
健康项已经进入并偶尔改变决策，不能说已经显著延寿。

DC-average每30秒强制轮换，三类负载下最大单堆损伤比Average高约143%到172%。它的三堆暴露
更均衡，但12次切换带来的工程代理损伤明显高于Average通常只有2次初始化启动的情况。该结果对
启停损伤口径敏感，当前只作为基线现象，不据此否定轮换策略。

## 5. 退化模型的边界

本轮先做了LZW总体暴露停止门筛查。618个有效段转移中，最佳负载模型相对单纯运行时钟的测试
RMSE增益为0；运行时长和电荷暴露相关系数为0.965，事件计数控制还优于最佳物理暴露模型。
结论是拒绝把它当作分动作在线退化率，只保留LZW健康终点和总体趋势。

Zuo在本轮只提供N+1系统思路和慢/快随机负载矩阵，不提供退化系数。现有退化代理能检查动作到
健康状态的闭环，但不能解释为已经从实车辨识出的真实物理退化率。

## 6. 当前问题

1. 180秒只够验证闭环和动作差异，不能回答EOL、RUL或寿命提升。
2. 健康目标在一半以上种子中没有改变结果，决策分辨率偏弱。
3. 最低16.57 kW负载在原25 A和90 A动作之间没有可跟踪点；本轮加入60 A IV插值动作。它不是
   LZW原始事件档位，后续需单独做动作网格敏感性。
4. DC-average结果被启停代理强烈影响，而启停、换堆和非计划停堆成本没有实车或台架标定。
5. 实车经验Markov负载是随机序列，不等于未参与建模的连续真实轨迹回放。
6. 40 kW仍是实车经验归一化参考，不是已经确认的铭牌额定净功率。

## 7. 现在看什么

主结果图：

`data/results/figures/fc_only_foundation/fig32_simplified_random_load_comparison.png`

正式结果：`data/results/simplified_random_load_comparison/`。先看`report.md`、
`aggregate_metrics.csv`和`paired_health_ablation.csv`。

LZW负面诊断：`data/results/lzw_overall_exposure_screen/`和Fig.31。

中文解释：

- `docs/SIMPLIFIED_DEGRADATION_RANDOM_LOAD_CHAIN_2026-07-14.md`
- `docs/RESULT_GUIDE_2026-07-14.md`

冒烟测试目录和额外的`current_core`重复图目录已删除。旧图保留为历史实验和否证证据，不作为当前
阅读入口。

## 8. 复现命令

```powershell
cd 'H:\其他\2026刘展玮\FC'
$env:PYTHONPATH='src'

python scripts\68_screen_lzw_overall_exposure.py
python scripts\69_run_simplified_random_load_comparison.py --length 180 --pair-seeds 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035 --jobs 12

python -m unittest discover -s tests -v
python scripts\run_baseline.py --mode check
python -m compileall -q src scripts tests
git diff --check
```

第二个实验在本机约需12分钟。脚本内置闭环验收，只要运行不完整、健康未逐步更新、在线堆数错误、
跟踪失败或出现约束违规，就会直接报错而不接受结果。

## 9. 下一步

下一阶段才进入效果优化。优先解决健康目标为何多数种子不改变决策，并把短时功率分配和长期运行堆
选择分开评估；同时补动作网格和启停代理敏感性。锂电池、未来需求预测和高级世界模型继续冻结，
直到当前退化感知基础方法的效果和适用边界稳定。
