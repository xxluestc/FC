# 多堆真实需求段公平基准

## 设置

- 冻结连续测试段长度：300 s
- Beam perfect-preview时域：16 s
- Beam终端SOC权重：300
- Rotating轮换周期：30 s
- 初始SOC：0.700
- 初始损伤：10% / 65% LZW late参考状态
- 需求裁剪点：3

## 公平性门槛

- 功率平衡、电池功率和SOC约束不可放宽且要求零违规；
- 驻留规则优先满足；急制动无可行动作时允许安全覆盖并单独计数；
- 主结论要求 `|SOC_end-SOC_ref|<=0.001`；
- 未达到门槛时仅报告原始指标和SOC等值氢耗筛查值，不做优劣结论。

当前满足末端SOC门槛的策略数：2/4。

## 结果

| strategy | n_steps | hydrogen_g | soc_equivalent_hydrogen_g | hydrogen_soc_corrected_g | degradation_increment_pct | performance_loss_sum | battery_throughput_kwh | soc_final | soc_error | switch_count | constraint_violation_steps | safety_override_steps | max_power_balance_error_kw | final_damage_mean_pct | final_damage_range_pct | stack_0_current_a_step | stack_1_current_a_step | runtime_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| average | 300 | 44.9662 | -1.52035 | 43.4458 | 0.00633878 | 4.33234 | 1.94952 | 0.700811 | 0.000810965 | 2 | 0 | 3 | 7.10543e-15 | 3.51056 | 5.14447 | 12660 | 12660 | 1.14743 |
| rotating | 300 | 54.5739 | -10.4514 | 44.1224 | 0.00610067 | 8.99475 | 1.70663 | 0.705575 | 0.00557488 | 2 | 0 | 4 | 7.10543e-15 | 3.51044 | 5.14433 | 10555 | 20175 | 6.48339 |
| instant_health | 300 | 54.4584 | -10.49 | 43.9685 | 0.00585428 | 5.21357 | 1.62791 | 0.705595 | 0.00559543 | 2 | 0 | 3 | 7.10543e-15 | 3.51031 | 5.1442 | 18575 | 12090 | 8.06843 |
| beam_perfect | 300 | 41.0236 | 1.3864 | 42.41 | 0.00603386 | 2.02513 | 1.67107 | 0.69926 | -0.000739516 | 2 | 0 | 1 | 7.10543e-15 | 3.5104 | 5.14453 | 15000 | 8100 | 212.81 |

## 严格SOC公平结果

Average与Beam满足 `|SOC_end-SOC_ref|<=0.001`，可作当前主比较。相对Average，Beam的SOC等值氢耗降低2.38%，累计Gamma损伤增量降低4.81%，性能损失累计值降低53.25%，电池吞吐降低14.28%。Beam将更多电流分配给初始更健康的0号堆，但300秒运行耗时约213秒，当前Python枚举实现尚不满足实时性。

Rotating与Instant未通过末端SOC门槛，只保留原始值和SOC等值筛查值，不用于上述优劣结论。3个需求点经过双堆+电池全局可行域裁剪，因此结果范围是当前可行域内的控制比较。
