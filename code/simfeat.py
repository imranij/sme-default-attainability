"""SaudiSME-Sim feature builder and the seven switchable sources.

The simulator itself is `lib/src/data/saudisme_sim.py`. This module only assembles its
output into the seven data sources the paper names, and implements the five selection
mechanisms of Section 5 of the paper, which the base simulator leaves as a single
incumbent-score rule.

EVERY quantity built here is simulator output and is labelled as such.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.saudisme_sim import simulate  # noqa: E402

SOURCES = ["ratios", "firmographics", "transactions", "balance", "einvoicing",
           "engagement", "calendar"]


def build(n_firms: int = 60_000, months: int = 24, seed: int = 0):
    static, seq = simulate(n_firms=n_firms, months=months, seed=seed)
    M = months
    arr = seq.drop(columns=["firm_id"]).to_numpy().reshape(len(static), M, 6)
    inflow, outflow, net, minbal, ntxn, einv = (arr[:, :, i] for i in range(6))

    def agg(a, pre):
        return {f"{pre}_mean": a.mean(1), f"{pre}_std": a.std(1),
                f"{pre}_min": a.min(1), f"{pre}_last6": a[:, -6:].mean(1)}

    cols = {}
    cols["ratios"] = {"age": static.age.to_numpy(), "leverage": static.leverage.to_numpy()}
    fg = pd.get_dummies(static[["sector", "size_band", "region"]], dtype=float)
    cols["firmographics"] = {c: fg[c].to_numpy() for c in fg.columns}
    tr = {}
    tr.update(agg(inflow, "inflow")); tr.update(agg(outflow, "outflow")); tr.update(agg(net, "net"))
    tr["net_ratio"] = net.mean(1) / np.maximum(inflow.mean(1), 1.0)
    cols["transactions"] = tr
    cols["balance"] = agg(minbal, "minbal")
    ei = agg(einv, "einv"); ei["einv_share"] = einv.mean(1) / np.maximum(inflow.mean(1), 1.0)
    cols["einvoicing"] = ei
    cols["engagement"] = agg(ntxn, "ntxn")
    # calendar source: month-of-year amplitude of the cash-flow series (the Hijri
    # seasonality the simulator inserts). Zero-cost source, per Section 5.
    amp = (inflow.max(1) - inflow.min(1)) / np.maximum(inflow.mean(1), 1.0)
    peak = inflow.argmax(1) % 12
    cols["calendar"] = {"seasonal_amplitude": amp, "peak_month": peak.astype(float),
                        "seasonal_ratio": inflow[:, :12].mean(1) / np.maximum(inflow[:, 12:].mean(1), 1.0)}

    frames = {s: pd.DataFrame(cols[s]) for s in SOURCES}
    X = pd.concat([frames[s] for s in SOURCES], axis=1)
    y = static.true_label.to_numpy().astype(float)
    return static, X, y, frames


def selection(static: pd.DataFrame, y: np.ndarray, mechanism: str, a0: float = 0.7, seed: int = 0):
    """Return a 0/1 approval flag at overall rate a0 under one of the five mechanisms."""
    rng = np.random.default_rng(1234 + seed)
    risk = static.incumbent_score.to_numpy()          # incumbent scorecard, higher = riskier
    n = len(risk)
    k = int(round(a0 * n))
    if mechanism == "score":
        idx = np.argsort(risk)[:k]
    elif mechanism == "score_plus_unobservable":
        # an unobservable correlated with default enters the decision: violates MAR and
        # the exclusion restriction a Heckman correction needs.
        u = 1.2 * y + rng.normal(0, 1.0, n)
        idx = np.argsort(risk + 0.8 * u)[:k]
    elif mechanism == "non_monotone":
        # a middle band is rejected by policy; the tails are approved.
        r = np.argsort(np.argsort(risk)) / n
        pref = np.where((r > 0.45) & (r < 0.75), 1.0, 0.0) + 0.001 * r
        idx = np.argsort(pref)[:k]
    elif mechanism == "human_override":
        base = np.argsort(risk)
        appr = np.zeros(n); appr[base[:k]] = 1
        flip = rng.random(n) < 0.12                   # 12% of files overridden at random
        appr[flip] = rng.integers(0, 2, flip.sum())
        # restore the overall rate
        need = k - int(appr.sum())
        pool = np.where(appr == 0)[0] if need > 0 else np.where(appr == 1)[0]
        pick = rng.choice(pool, abs(need), replace=False) if abs(need) <= len(pool) else pool
        appr[pick] = 1 if need > 0 else 0
        return appr.astype(int)
    elif mechanism == "policy_floor":
        # sector/region floors: every firm in two sectors is approved before ranking.
        floor = static.sector.isin(["construction", "manufacturing"]).to_numpy()
        pref = risk - 100.0 * floor
        idx = np.argsort(pref)[:k]
    else:
        raise ValueError(mechanism)
    appr = np.zeros(n, int)
    appr[idx] = 1
    return appr
