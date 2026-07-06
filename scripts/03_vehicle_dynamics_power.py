"""构建实测需求功率，并生成独立车辆动力学基线。

中文名：03_构建需求功率与车辆动力学。动力学只提供物理趋势对照。
"""

from pathlib import Path
import argparse, json, sys
import numpy as np, pandas as pd
from scipy.signal import savgol_filter
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fc_power.vehicle_dynamics import VehicleParams, force_power


def main():
    """片段内平滑车速/求加速度，分牵引制动校准动力学功率。"""
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--metrics", type=Path, required=True)
    a = p.parse_args()
    d = pd.read_csv(a.input, parse_dates=["timestamp"])
    sm = np.empty(len(d))
    acc = np.empty(len(d))
    sm[:] = np.nan
    acc[:] = np.nan
    # 每个segment独立处理，禁止跨停车、跨夜或数据断点做平滑/差分。
    for _, ix in d.groupby("segment_id").groups.items():
        ix = np.asarray(list(ix))
        v = d.loc[ix, "speed_mps"].interpolate().to_numpy()
        win = min(11, len(v) if len(v) % 2 else len(v) - 1)
        vs = savgol_filter(v, win, 2) if win >= 5 else v
        sm[ix] = vs
        acc[ix] = np.gradient(vs, 1.0)
    d["speed_smooth_mps"] = sm
    d["acceleration_smooth_mps2"] = acc
    comp = force_power(sm, acc, VehicleParams())
    for k, v in comp.items():
        d[k] = v
    # 当前实测需求定义为电机V×I/1000，保留原传感器充放电符号。
    y = d.motor_power_kw_raw_sign.to_numpy()
    x = d.p_wheel_kw.to_numpy()
    valid = np.isfinite(x) & np.isfinite(y) & (abs(y) < 300) & (abs(acc) < 4)
    split = int(0.7 * len(d))
    train = valid & (np.arange(len(d)) < split)
    test = valid & (np.arange(len(d)) >= split)
    # Separate traction and regenerative calibration; fit intercepts/slopes robustly.
    pred = np.full(len(d), np.nan)
    models = {}
    for name, mask in [("traction", x >= 0), ("braking", x < 0)]:
        tr = train & mask
        model = HuberRegressor(epsilon=1.35).fit(x[tr, None], y[tr])
        pred[mask] = model.predict(x[mask, None])
        models[name] = {
            "intercept_kw": float(model.intercept_),
            "slope": float(model.coef_[0]),
            "train_n": int(tr.sum()),
        }
    d["p_dem_dyn_calibrated_kw"] = pred
    d["p_dem_measured_kw"] = y
    d["dynamics_residual_kw"] = y - pred

    def met(mask):
        return {
            "n": int(mask.sum()),
            "mae_kw": mean_absolute_error(y[mask], pred[mask]),
            "rmse_kw": mean_squared_error(y[mask], pred[mask]) ** 0.5,
            "r2": r2_score(y[mask], pred[mask]),
            "energy_bias_kwh": float(np.sum(pred[mask] - y[mask]) / 3600),
        }

    metrics = {
        "split": "first 70% calibration, final 30% test",
        "models": models,
        "test_all": met(test),
        "test_traction": met(test & (x >= 0)),
        "test_braking": met(test & (x < 0)),
        "vehicle_params": VehicleParams().__dict__,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.metrics.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(a.output, index=False)
    a.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
