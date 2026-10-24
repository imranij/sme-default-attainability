"""SaudiSME-Sim: synthetic SME cash-flow and default generator.

Generates firms (sector x size x region), 24 monthly cash-flow vectors with Hijri
seasonality, an incumbent-scorecard reject mechanism, and TRUE labels for all firms,
so that selection effects can be evaluated exactly.

The sector and size shares below are illustrative round numbers chosen to give a plausible
SME mix. They are not taken from a published statistical release, and no Saudi firm-level
data enter the simulator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SECTOR_SHARES = {"wholesale_retail": 0.32, "construction": 0.18, "manufacturing": 0.10,
                 "accommodation_food": 0.09, "professional_services": 0.08, "transport": 0.06,
                 "health_education": 0.05, "other": 0.12}
SIZE_SHARES = {"micro": 0.86, "small": 0.11, "medium": 0.03}
REGIONS = ["Riyadh", "Makkah", "Eastern", "Madinah", "Qassim", "Asir", "Other"]
# multiplicative seasonal factors by Hijri month (1..12), sector-specific (illustrative)
HIJRI_SEASON = {
    "wholesale_retail": {9: 1.35, 10: 1.20, 12: 1.10},        # Ramadan, Shawwal(Eid), Dhul-Hijjah
    "accommodation_food": {9: 0.80, 10: 1.25, 12: 1.40},      # Ramadan dip, Eid + Hajj peaks
    "construction": {9: 0.85, 10: 0.90},
}


def gregorian_to_hijri_month(dates: pd.DatetimeIndex) -> np.ndarray:
    try:
        from hijridate import Gregorian  # optional dependency; the fallback below is used without it
        return np.array([Gregorian(d.year, d.month, d.day).to_hijri().month for d in dates])
    except ImportError:  # crude fallback: 354.37-day lunar year from a known epoch
        epoch = pd.Timestamp("2023-07-19")  # 1 Muharram 1445 (approx.)
        days = (dates - epoch).days.values
        return ((days % 354.37) // 29.53).astype(int) % 12 + 1


def simulate(n_firms: int = 100_000, months: int = 24, start: str = "2022-01-01",
             incumbent_approval_rate: float = 0.7, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (static_df, seq_df). static_df has true_label, approved_flag, observed_label."""
    rng = np.random.default_rng(seed)
    sectors = rng.choice(list(SECTOR_SHARES), n_firms, p=list(SECTOR_SHARES.values()))
    sizes = rng.choice(list(SIZE_SHARES), n_firms, p=list(SIZE_SHARES.values()))
    regions = rng.choice(REGIONS, n_firms)
    age = rng.gamma(2.0, 3.0, n_firms)
    leverage = np.clip(rng.beta(2, 5, n_firms) * 2, 0, 2)
    base_rev = np.exp(rng.normal({"micro": 10.5, "small": 12.5, "medium": 14.0}[sizes[0]], 0.6, n_firms))

    dates = pd.date_range(start, periods=months, freq="MS")
    hm = gregorian_to_hijri_month(dates)
    seq = np.zeros((n_firms, months, 6))  # inflow, outflow, net, min_balance, n_txn, einvoice
    vol = rng.uniform(0.05, 0.4, n_firms)
    ar = np.zeros(n_firms)
    for t in range(months):
        ar = 0.7 * ar + rng.normal(0, vol)
        season = np.array([HIJRI_SEASON.get(s, {}).get(int(hm[t]), 1.0) for s in sectors])
        inflow = base_rev * season * np.exp(ar)
        outflow = inflow * rng.uniform(0.8, 1.05, n_firms) * (1 + 0.1 * leverage)
        seq[:, t, 0] = inflow
        seq[:, t, 1] = outflow
        seq[:, t, 2] = inflow - outflow
        seq[:, t, 3] = np.minimum(0, inflow - outflow) * rng.uniform(0.5, 1.5, n_firms)
        seq[:, t, 4] = rng.poisson(20 + inflow / 5000)
        seq[:, t, 5] = inflow * rng.uniform(0.6, 1.0, n_firms)  # e-invoice turnover proxy

    z = (-0.15 * age + 1.2 * leverage + 2.5 * vol
         - 0.8 * (seq[:, :, 2].mean(1) / base_rev) + rng.normal(0, 0.5, n_firms))
    p_default = 1 / (1 + np.exp(-(z - 2.2)))
    true_label = (rng.random(n_firms) < p_default).astype(int)
    incumbent = leverage - 0.05 * age + rng.normal(0, 0.8, n_firms)   # ratio-only scorecard (risk)
    tau = np.quantile(incumbent, incumbent_approval_rate)
    approved = (incumbent <= tau).astype(int)

    static = pd.DataFrame({"firm_id": np.arange(n_firms), "sector": sectors, "size_band": sizes,
                           "region": regions, "age": age, "leverage": leverage,
                           "incumbent_score": incumbent, "approved_flag": approved,
                           "true_label": true_label,
                           "observed_label": np.where(approved == 1, true_label, np.nan)})
    seq_df = pd.DataFrame(seq.reshape(n_firms, -1),
                          columns=[f"m{t:02d}_{c}" for t in range(months)
                                   for c in ["inflow", "outflow", "net", "minbal", "ntxn", "einv"]])
    seq_df.insert(0, "firm_id", static.firm_id)
    return static, seq_df


if __name__ == "__main__":
    s, q = simulate(5000)
    print(s.true_label.mean(), s.loc[s.approved_flag == 1, "true_label"].mean())
