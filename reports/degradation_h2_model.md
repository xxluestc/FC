# Phase 5 电堆退化代理与氢耗

电流档位和性能损失成本来自刘占伟 `theta=[i0, ih, R_ohm]` 与电压模型生成的 Stage 6 电流点表，不采用韩成杰或其他论文的退化系数。该量表示“当前老化状态下相对健康基线的等效性能损失”，不是动作导致的真实材料退化增量。

氢耗第一版使用法拉第理论式：`m_H2=N_cell I M_H2/(2F)`，N_cell=170。它是可追溯理论基线；刘占伟半月CSV包含累计/单次耗氢字段，后续只有通过分辨率、重置和单调性审计后才能用于校准。

输出 `data/processed/current_point_degradation_h2.csv` 包含实际数据中的电流档位、功率、raw/clipped/normalized退化代理和理论氢耗率。严禁把 normalized cost 解释成 mV/s 或寿命百分比。

