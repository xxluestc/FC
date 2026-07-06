"""由刘占伟性能损失表生成电流档位、退化代理和理论氢耗。

中文名：02_构建电堆退化代理与氢耗表。输出proxy不是真实材料退化系数。
"""

from pathlib import Path
import argparse, sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fc_power.hydrogen_model import faraday_h2_g_s


def main():
    """优先取late健康状态，并按170节电堆计算功率和法拉第氢耗。"""
    p = argparse.ArgumentParser()
    p.add_argument("--liu-cost-table", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    d = pd.read_csv(a.liu_cost_table)
    # 固定late状态形成一张可调用档位表；不在baseline内搜索健康状态。
    if "health_state" in d:
        d = d[d.health_state.eq("late")].copy()
    out = pd.DataFrame(
        {
            "current_a": d.current_A,
            "stack_power_kw": d.current_A * d.V_aged_cell_V * 170 / 1000,
            "aged_cell_voltage_v": d.V_aged_cell_V,
            "performance_loss_cost_raw_wh_step": d.equivalent_energy_loss_raw_Wh_per_1s,
            "performance_loss_cost_clipped_wh_step": d.equivalent_energy_loss_clipped_Wh_per_1s,
            "performance_loss_cost_normalized": d.normalized_energy_cost_0_1,
            "faraday_h2_g_s": faraday_h2_g_s(d.current_A, 170),
        }
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.output, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
