# 健康观测预测/校正接口

更新时间：2026-07-13

## 1. 定位

本接口关闭的是“代码没有观测接入口”这一工程缺口，不代表已经取得21UBE0022同车同堆健康
posterior。当前真实主实验仍只使用动作驱动预测态。身份审计已经证明旧MAT无法与2025--2026
车辆流逐行绑定；长期电压又受到两个时期台阶和运行条件混杂，因此都不能直接回写累计退化状态。

## 2. 因果时序

```text
当前posterior状态
  -> 规划器只调用MechanisticMultiStackWorldModel.step做候选rollout
  -> 选择并执行一个动作
  -> 条件均值健康prediction
  -> 接收同一执行步末端观测
  -> HealthObserver.correct
  -> posterior写入下一次决策状态
```

`ObservedHealthExecutionLoop.execute`只用于已选动作。它不替换世界模型的`step`，因此Beam、Instant
和安全投影不能在候选枚举时读取未来观测。observer本身没有可变内部状态，均值、方差、校正次数
和观测来源全部由`HealthBelief`显式携带。

## 3. 当前实现

- `DegradationObservation`：带时间戳、方差、来源和synthetic标志的标量退化代理观测；
- `GaussianDegradationObserver.predict`：传播条件均值与方差；连续Gamma通道使用
  $\operatorname{Var}[\Delta D]=E[\Delta D]b$，事件通道保持确定性；
- `GaussianDegradationObserver.correct`：标量高斯更新，保留innovation、gain、raw posterior和投影记录；
- 单调投影：校正后累计退化不得低于该执行步开始时的累计值，但允许纠正本步预测增量的高估；
- `ObservedHealthExecutionLoop`：逐堆独立校正，只把posterior写入下一步状态；当前步功率、成本和约束
  仍保留执行前模型prediction，避免事后观测改写已经发生的物理输出。

主要代码：

- `src/fc_power/health/observer.py`
- `src/fc_power/world_model/observed_health.py`
- `tests/test_health_observer.py`

## 4. 合成接口验证

`scripts/52_validate_synthetic_health_observer.py`使用冻结LZW/Ghaderi参数构造720小时合成链。真值采用
预声明1.15异质性条件均值，名义模型采用1.0，每24小时输入一次标准差0.03个百分点的直接退化代理：

| 指标 | 数值 |
|---|---:|
| 观测次数 | 30 |
| 开环RMSE | 0.122200 %-point |
| 校正belief RMSE | 0.022866 %-point |
| 合成RMSE降低 | 81.288% |
| 末端开环绝对误差 | 0.211470 %-point |
| 末端posterior绝对误差 | 0.008184 %-point |
| 单调投影次数 | 16 |

结果位于`data/results/synthetic_health_observer/`，Fig.17为320 DPI PNG。该实验只验证接口时序、
漂移校正和审计字段，不证明真实observer精度；100%合成区间覆盖只说明当前假设下区间保守，不能
解释为实车校准结果。

## 5. 真实观测审计结果与接入契约

刘占伟论文和源码确认旧MAT来自2022--2024、170节、约3000小时的长期电堆链；`21UBE0022`
七天负载则属于2025年车辆数据。两者没有堆编号和连续时间证明，旧MAT只保留为跨数据集先验。

对`G:`长期车辆流的总堆电压做固定电流和65--70 C温度匹配后，共保留2,635,767行、942个
日档观测，并识别出2025-06和2025-08两个跨电流档台阶。2025-08至2026-06的原始匹配趋势为
-1.984%/100天；继续校正实际电流、进出口温度和氢/空气压力后为-0.182%/100天，95%区间
[-0.639%, 0.060%]，包含0。因此当前电压只作执行后性能残差监测，不转成`D`观测。

未来若启用真实校正，仍必须提供：

真实适配器必须先提供：

1. 车辆ID、堆ID和与21UBE0022可核查的来源关系；
2. 电流、电压、温度、压力等变量名、单位、采样率、滤波和时间戳；
3. 从原始传感量到`theta=[i0, ih, R_ohm]`或$D$的UKF/PF/RLS测量模型；
4. prediction与observation的时间对齐规则、缺测策略和观测噪声协方差；
5. 独立连续段上的innovation、posterior误差、区间覆盖和漂移诊断。

在上述契约满足前，不允许把合成`DegradationObservation`替换成未经身份和单位审计的`x_est`行，
也不把环境混杂的电压趋势硬映射成`D`。详细证据见
`docs/DATA_IDENTITY_POWER_HEALTH_AUDIT_2026-07-13.md`。
