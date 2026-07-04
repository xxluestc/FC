import numpy as np
from fc_power.battery_model import next_soc, throughput_cost

DEFAULT_WEIGHTS = {
    "hydrogen": 0.45,
    "degradation_proxy": 1.0,
    "battery_use": 1.5,
    "soc": 3.0,
    "switch": 0.08,
    "smooth": 0.005,
}


def choose(
    preview,
    soc,
    prev,
    dwell,
    actions,
    h2,
    deg,
    beam_width=8,
    min_dwell=15,
    soc_ref=0.70,
    weights=None,
):
    weights = DEFAULT_WEIGHTS if weights is None else weights
    beam = [(0.0, soc, prev, dwell, prev)]
    for j, pdem in enumerate(preview):
        cand = []
        for cost, s, t, dw, first in beam:
            allowed = (
                [t]
                if dw < min_dwell
                else range(max(0, t - 1), min(len(actions), t + 2))
            )
            for nt in allowed:
                pfc = actions[nt]
                pbat = pdem - pfc
                sn = float(next_soc(s, pbat))
                if not (-75 <= pbat <= 120 and 0.30 <= sn <= 0.90):
                    continue
                # Charge-sustaining ECMS-like feedback: a SOC deficit raises
                # the desired FC contribution before the short horizon can see
                # the end of the trip.
                pref = np.clip(
                    max(pdem, 0) + 1200 * (soc_ref - s), actions.min(), actions.max()
                )
                c = (
                    weights["hydrogen"] * h2[nt] / max(h2.max(), 1e-9)
                    + weights["degradation_proxy"] * deg[nt]
                    + weights["battery_use"] * abs(pbat) / 120
                    + weights["soc"] * abs(pfc - pref) / max(actions.max(), 1)
                    + 0.5 * abs(sn - soc_ref) / 0.1
                    + weights["switch"] * (nt != t)
                    + weights["smooth"]
                    * abs(pfc - actions[t])
                    / max(np.diff(actions).max(), 1)
                )
                nd = (
                    min_dwell
                    if nt == t and dw >= min_dwell
                    else dw + 1 if nt == t else 1
                )
                cand.append((cost + c, sn, nt, nd, nt if j == 0 else first))
        if not cand:
            return prev
        beam = sorted(
            cand, key=lambda z: z[0] + 50 * max(0, abs(z[1] - soc_ref) - 0.02) ** 2
        )[:beam_width]
    return min(beam, key=lambda z: z[0] + 50 * max(0, abs(z[1] - soc_ref) - 0.02) ** 2)[
        4
    ]
