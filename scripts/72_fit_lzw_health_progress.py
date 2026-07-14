"""Fit and audit the coefficient-free LZW theta progress coordinate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fc_power.health import fit_lzw_health_progress
from fc_power.health.lzw_health_progress import THETA_COLUMNS


ROOT = Path(__file__).resolve().parents[1]
THETA_SOURCE = ROOT / "data/upstream_lzw/theta_event_trajectory_6104.csv"
EVENT_SOURCE = ROOT / "data/upstream_lzw/canonical_event_table_6104.csv"
OUTPUT = ROOT / "data/results/lzw_health_progress"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plot_diagnostics(
    frame: pd.DataFrame,
    mapping,
    h: np.ndarray,
    output_path: Path,
) -> None:
    colors = ("#0072B2", "#D55E00", "#009E73")
    labels = (r"$i_0$", r"$i_h$", r"$R$")
    theta_axis_labels = (
        r"$i_0$ (A cm$^{-2}$)",
        r"$i_h$ (A cm$^{-2}$)",
        r"$R$ ($\Omega$ cm$^2$)",
    )
    observed = frame.loc[:, THETA_COLUMNS].to_numpy(dtype=float)
    reconstructed = mapping.theta_at(h)
    component_h = mapping.component_progress(observed)
    event_index = np.arange(1, len(frame) + 1)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.3), constrained_layout=True)
    axis = axes[0, 0]
    for index, (color, label) in enumerate(zip(colors, labels)):
        axis.plot(
            event_index,
            component_h[:, index],
            color=color,
            alpha=0.55,
            lw=0.8,
            label=f"{label} component",
        )
    axis.plot(event_index, h, color="#222222", lw=1.6, label="Aggregate h")
    axis.set(xlabel="Ordered LZW event", ylabel="Normalized progress")
    axis.set_title("(a) Coefficient-free progress coordinate", loc="left")
    axis.set_ylim(-0.03, 1.03)
    axis.legend(frameon=False, ncol=2)

    for index, axis in enumerate((axes[0, 1], axes[1, 0], axes[1, 1])):
        stride = max(1, len(h) // 900)
        axis.scatter(
            h[::stride],
            observed[::stride, index],
            s=5,
            color=colors[index],
            alpha=0.32,
            edgecolors="none",
            label="UKF-PF trajectory",
        )
        axis.plot(h, reconstructed[:, index], color="#222222", label="Monotone map")
        axis.set(xlabel="Progress h", ylabel=theta_axis_labels[index])
        axis.set_title(
            f"({chr(ord('b') + index)}) {labels[index]} reconstruction",
            loc="left",
        )
        axis.legend(frameon=False)

    for axis in axes.ravel():
        axis.grid(True, color="#D9D9D9", lw=0.5, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    theta = pd.read_csv(THETA_SOURCE)
    event_keys = pd.read_csv(
        EVENT_SOURCE,
        usecols=["event_id", "canonical_row_6104", "original_index"],
    )
    mapping, h, diagnostics = fit_lzw_health_progress(
        theta,
        endpoint_window=50,
        key_reference=event_keys,
    )
    reconstructed = mapping.theta_at(h)
    result = theta.loc[
        :, ["event_id", "canonical_row_6104", "original_index", *THETA_COLUMNS]
    ].copy()
    result.insert(3, "health_progress_h", h)
    for index, column in enumerate(THETA_COLUMNS):
        result[f"{column}_monotone"] = reconstructed[:, index]

    diagnostics["interpretation"] = {
        "accepted": (
            "h is a normalized coordinate along the recorded LZW theta trajectory; "
            "theta(h) is a descriptive monotone performance manifold"
        ),
        "prohibited": (
            "h is not SOH, h=1 is not EOL, and the reconstruction is not an "
            "independent prediction or an action-resolved degradation law"
        ),
        "controller_changed": False,
        "uses_damage_proxy_D": False,
        "uses_gamma_or_literature_action_coefficients": False,
    }
    diagnostics["source_rows"] = {
        "theta": int(len(theta)),
        "event_keys": int(len(event_keys)),
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT / "health_progress_trajectory.csv", index=False)
    (OUTPUT / "mapping.json").write_text(
        json.dumps(mapping.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT / "diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "theta_source": str(THETA_SOURCE.relative_to(ROOT)),
        "theta_sha256": sha256(THETA_SOURCE),
        "event_source": str(EVENT_SOURCE.relative_to(ROOT)),
        "event_sha256": sha256(EVENT_SOURCE),
        "output_scope": "cross-dataset LZW health-progress prior",
    }
    (OUTPUT / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_diagnostics(result, mapping, h, OUTPUT / "fig34_lzw_health_progress.png")

    component_lines = []
    for column in THETA_COLUMNS:
        item = diagnostics["components"][column]
        component_lines.append(
            f"| `{column}` | {item['degradation_aligned_spearman_vs_row']:.6f} "
            f"| {item['reconstruction']['normalized_rmse']:.6f} |"
        )
    report = """# LZW health-progress gate H0

## Decision

`GO` for a descriptive `theta -> h -> theta(h)` manifold.  `NO-GO` for an
action-resolved degradation transition or controller integration.

The fit uses only the ordered LZW UKF-PF theta trajectory.  It does not read
the cumulative damage proxy `D`, Gamma parameters, or Ghaderi/Pei action
coefficients.  `h=1` denotes only the endpoint of this recorded trajectory; it
does not denote SOH=0, EOL, failure, or a known RUL boundary.

## Diagnostics

| Component | Degradation-aligned Spearman vs row | In-sample normalized RMSE |
|---|---:|---:|
""" + "\n".join(component_lines) + """

The reconstruction error is descriptive and in-sample because `h` is built
from the same three theta components.  It verifies numerical consistency of
the manifold, not predictive validity.  The older MAT identity remains
separate from vehicle `21UBE0022`.
"""
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
