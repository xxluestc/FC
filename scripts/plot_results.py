from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fc_power.evaluation.plots import paper_style

paper_style(plt)
F = Path("figures")
F.mkdir(exist_ok=True)
C = {
    "Measured": "#333333",
    "Dynamics": "#0072B2",
    "Predicted": "#D55E00",
    "Perfect": "#009E73",
}


def save(fig, n):
    fig.tight_layout()
    fig.savefig(F / f"{n}.png")
    fig.savefig(F / f"{n}.pdf")
    plt.close(fig)


d = pd.read_csv("data/processed/power_demand_from_dynamics.csv")
z = d.iloc[-5000:-4000]
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(
    np.arange(len(z)) / 60,
    z.p_dem_measured_kw,
    c=C["Measured"],
    lw=0.7,
    label="Measured motor demand",
)
ax.plot(
    np.arange(len(z)) / 60,
    z.p_dem_dyn_calibrated_kw,
    c=C["Dynamics"],
    lw=0.8,
    label="Calibrated dynamics",
)
ax.set(xlabel="Time (min)", ylabel="Power (kW)")
ax.legend(frameon=False, ncol=2)
ax.grid(alpha=0.18)
save(fig, "vehicle_dynamics_fit")
m = pd.read_csv("data/results/prediction_metrics.csv")
methods = ["speed_only_dynamics", "state_direct_power", "hybrid_physics_corrected"]
labels = ["Speed-only dynamics", "State direct power", "Physics + residual"]
colors = ["#0072B2", "#009E73", "#D55E00"]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3))
for ax, metric, ylabel in [
    (axs[0], "speed_rmse_mps", "Speed RMSE (m/s)"),
    (axs[1], "nrmse_range_pct", "Mean-power NRMSE (% full scale)"),
]:
    for name, label, c in zip(methods, labels, colors):
        q = m[m.method.eq(name)]
        ax.plot(q.horizon_s, q[metric], marker="o", ms=3, label=label, c=c)
    ax.set(xlabel="Prediction horizon (s)", ylabel=ylabel, xticks=[1, 3, 5, 10, 15])
    ax.grid(alpha=0.18)
axs[0].legend(frameon=False, fontsize=7)
save(fig, "prediction_horizon_errors")
pr = pd.read_csv("data/processed/prediction_results.csv")
origin = int(pr.origin_index.quantile(0.4))
q = pr[(pr.origin_index == origin) & (pr.method == "state_direct_power")]
fig, axs = plt.subplots(2, 1, figsize=(6.8, 4.5), sharex=True)
axs[0].plot(
    q.horizon_s, q.speed_actual_mps * 3.6, c=C["Measured"], marker="o", label="Actual"
)
axs[0].plot(
    q.horizon_s, q.speed_pred_mps * 3.6, c=C["Predicted"], marker="s", label="Predicted"
)
axs[0].set_ylabel("Speed (km/h)")
axs[0].legend(frameon=False)
axs[1].plot(q.horizon_s, q.power_actual_kw, c=C["Measured"], marker="o")
axs[1].plot(q.horizon_s, q.power_pred_kw, c=C["Predicted"], marker="s")
axs[1].set(xlabel="Horizon (s)", ylabel="Demand power (kW)")
[a.grid(alpha=0.18) for a in axs]
save(fig, "pred_speed_power_compare")
tr = pd.read_csv("data/results/allocation/allocation_trajectory.csv")
fig, axs = plt.subplots(2, 1, figsize=(7.2, 5), sharex=True)
for name, c in [
    ("instant", "#777777"),
    ("constant", "#E69F00"),
    ("perfect", "#009E73"),
    ("predicted", "#0072B2"),
]:
    q = tr[tr.strategy.eq(name)]
    if name != "instant":
        q = q[q.horizon_s.eq(5)]
    axs[0].plot(q.step / 60, q.p_fc_kw, lw=0.8, label=name, c=c)
    axs[1].plot(q.step / 60, q.soc, lw=0.9, label=name, c=c)
axs[0].set_ylabel("FC power (kW)")
axs[0].legend(frameon=False, ncol=4)
axs[1].axhspan(0.68, 0.72, color="#009E73", alpha=0.08)
axs[1].set(xlabel="Time (min)", ylabel="SOC")
[a.grid(alpha=0.18) for a in axs]
save(fig, "power_split_soc_compare")
