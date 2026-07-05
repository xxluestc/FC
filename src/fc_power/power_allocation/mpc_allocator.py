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
    block_size=1,
    max_horizon_switches=None,
):
    weights = DEFAULT_WEIGHTS if weights is None else weights
    beam = [(0.0, soc, prev, dwell, prev, 0)]
    for j, pdem in enumerate(preview):
        cand = []
        for cost, s, t, dw, first, horizon_switches in beam:
            must_hold = (
                dw < min_dwell
                or (j > 0 and block_size > 1 and j % block_size != 0)
                or (
                    max_horizon_switches is not None
                    and horizon_switches >= max_horizon_switches
                )
            )
            allowed = (
                [t] if must_hold else range(max(0, t - 1), min(len(actions), t + 2))
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
                cand.append(
                    (
                        cost + c,
                        sn,
                        nt,
                        nd,
                        nt if j == 0 else first,
                        horizon_switches + int(nt != t),
                    )
                )
        if not cand:
            return prev
        beam = sorted(
            cand, key=lambda z: z[0] + 50 * max(0, abs(z[1] - soc_ref) - 0.02) ** 2
        )[:beam_width]
    return min(beam, key=lambda z: z[0] + 50 * max(0, abs(z[1] - soc_ref) - 0.02) ** 2)[
        4
    ]
