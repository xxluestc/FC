# 长期Gamma不确定性敏感性

- 等效暴露：1000 h
- Monte Carlo样本：5000
- 终点CV假设：[0.05, 0.1, 0.2]
- 连续负载Gamma增量按可加性聚合；启停/变载保持确定性事件增量。
- 短周期重复只用于放大并比较策略暴露，不代表真实车辆长期重复同一段工况。

## 策略配对结果

| 场景 | CV | P(Beam退化更低) | Beam-Average均值(%) | 差值P95(%) |
|---|---:|---:|---:|---:|
| real_block_bootstrap_seed_2026 | 5% | 100.0% | -9.232 | -8.996 |
| real_block_bootstrap_seed_2027 | 5% | 100.0% | -1.885 | -1.853 |
| synthetic_event_markov_seed_2026 | 5% | 0.0% | 0.601 | 0.670 |
| synthetic_event_markov_seed_2027 | 5% | 100.0% | -4.565 | -4.533 |
| real_block_bootstrap_seed_2026 | 10% | 100.0% | -9.235 | -8.768 |
| real_block_bootstrap_seed_2027 | 10% | 100.0% | -1.885 | -1.822 |
| synthetic_event_markov_seed_2026 | 10% | 0.0% | 0.601 | 0.739 |
| synthetic_event_markov_seed_2027 | 10% | 100.0% | -4.565 | -4.500 |
| real_block_bootstrap_seed_2026 | 20% | 100.0% | -9.240 | -8.288 |
| real_block_bootstrap_seed_2027 | 20% | 100.0% | -1.886 | -1.761 |
| synthetic_event_markov_seed_2026 | 20% | 0.0% | 0.603 | 0.889 |
| synthetic_event_markov_seed_2027 | 20% | 100.0% | -4.565 | -4.431 |

## 可解释结论

- 事件型确定性损伤占总期望增量的96.9%–98.2%，当前长期排序主要由启停/变载等事件系数和策略事件次数决定，而不是Gamma连续增量方差。
- CV从5%增至20%会扩大分布宽度，但没有改变本次场景的策略排序。
- Beam并非普遍占优；出现Beam总退化更高的场景：synthetic_event_markov_seed_2026。因此后续不能用单个随机种子声称延寿。
- 本实验保留Gamma作为不可逆在线状态及边界不确定性；主控制比较使用条件均值，CV只进入敏感性分析。
