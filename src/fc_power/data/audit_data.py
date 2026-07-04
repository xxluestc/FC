"""Lightweight, non-mutating audit helpers for heterogeneous research data."""

from __future__ import annotations
from pathlib import Path
from scipy.io import whosmat
import pandas as pd

SIGNALS = {
    "timestamp": ("时间", "time", "timestamp", "date"),
    "speed": ("车速", "speed", "velocity"),
    "demand_power": ("目标功率", "需求功率", "power_req", "p_dem"),
    "fc_power": ("燃料电池输出功率", "DCDC输入", "fc_power"),
    "battery_power": ("电池电压", "电池电流", "battery_power"),
    "soc": ("SOC", "soc"),
    "current": ("电流", "current"),
    "voltage": ("电压", "voltage"),
}


def csv_header(path: Path) -> list[str]:
    last = None
    for enc in ("utf-8-sig", "gb18030", "utf-8"):
        try:
            return list(pd.read_csv(path, nrows=0, encoding=enc).columns.astype(str))
        except Exception as exc:
            last = exc
    raise last


def match_signals(columns: list[str]) -> dict[str, list[str]]:
    return {
        key: [c for c in columns if any(token.lower() in c.lower() for token in tokens)]
        for key, tokens in SIGNALS.items()
    }


def inventory(root: Path) -> dict[str, int]:
    return {
        ext: sum(1 for _ in root.rglob(f"*{ext}"))
        for ext in (".csv", ".xlsx", ".mat", ".m", ".py")
    }


def mat_inventory(root: Path) -> list[dict]:
    out = []
    for f in root.rglob("*.mat"):
        try:
            variables = [
                {"name": n, "shape": list(s), "class": c} for n, s, c in whosmat(f)
            ]
        except Exception as exc:
            variables = [{"error": str(exc)}]
        out.append({"path": str(f), "variables": variables})
    return out
