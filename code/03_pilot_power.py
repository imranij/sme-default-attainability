"""Q4 pilot power surfaces on the simulator (the pilot-size analysis of the paper).

Monte Carlo, at a true reduction of 15%, of the probability of rejecting the null of
no reduction at the 5% level as a function of cohort size n and duration in months,
under the three designs of Section 7.5:
  (i)   observational  -- accepted loans only; the estimand is the identified set of
        Proposition 2, so the test is on its LOWER endpoint at the true M;
  (ii)  randomised band -- a share of applicants near the threshold approved at random,
        which identifies mu_out inside the band and shrinks the unidentified share;
  (iii) full randomisation of the swap set -- DIR is point-identified.

The base rate, approval rate, swap share and true risk ratio are taken from the
simulator run of 02_selection_mechanisms.py (mechanism "score"); monthly
autocorrelation enters as a design effect. Every number here is simulator output.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from common import RESULTS, dump

A = 0.7
TRUE_R = 0.15
RHO_MONTH = 0.3          # monthly autocorrelation of the cohort default rate
BAND_SHARE = 0.10        # share of applicants randomised in design (ii)
N_GRID = [2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000]
MONTH_GRID = [6, 12, 18, 24]
REPS = 400


def load_params():
    p = os.path.join(RESULTS, "Q2_sim_mechanisms.json")
    rows = json.load(open(p))
    sel = [r for r in rows if r["mechanism"] == "score"]
    DI0 = float(np.mean([r["DI0"] for r in sel]))
    s = float(np.mean([r["swap_share"] for r in sel]))
    M = float(np.mean([r["true_M"] for r in sel]))
    return DI0, s, M


def power_cell(n, months, DI0, s, M, design, rng):
    """Monte-Carlo rejection rate at a true 15% reduction."""
    n_appr = int(round(A * n))
    n_in = int(round((1 - s) * n_appr))
    n_out = n_appr - n_in
    if n_in < 5 or n_out < 2:
        return 0.0
    # true rates consistent with DIR = 0.15 at the true risk ratio M
    di_new = (1 - TRUE_R) * DI0
    mu_in = di_new / (1 + s * (M - 1))
    mu_out = M * mu_in
    deff = 1.0 + (months - 1) * RHO_MONTH          # design effect from monthly autocorrelation

    def draw(k, p):
        """Over-dispersed binomial count: variance inflated by the design effect."""
        if k <= 0:
            return np.zeros(REPS)
        var = k * p * (1 - p) * deff
        return np.clip(rng.normal(k * p, np.sqrt(max(var, 1e-9)), REPS), 0, k)

    d_in = draw(n_in, mu_in)
    d_old = draw(n_appr, DI0)
    hat_mu_in = d_in / n_in
    hat_DI0 = d_old / n_appr
    se_in = np.sqrt(np.maximum(hat_mu_in * (1 - hat_mu_in) / n_in, 1e-12) * deff)
    se_0 = np.sqrt(np.maximum(hat_DI0 * (1 - hat_DI0) / n_appr, 1e-12) * deff)

    if design == "observational":
        # lower endpoint of the identified set at the true M; reject "no reduction" if
        # its one-sided 95% lower confidence limit exceeds 0.
        eff = 1 + s * (M - 1)
        lo = 1 - eff * hat_mu_in / hat_DI0
        se = eff * hat_mu_in / hat_DI0 * np.sqrt((se_in / hat_mu_in) ** 2 + (se_0 / hat_DI0) ** 2)
    elif design == "randomised_band":
        # mu_out is observed for the randomised share of the swap set; the residual
        # unidentified share is s*(1 - BAND_SHARE/s) when BAND_SHARE < s, else 0.
        covered = min(1.0, BAND_SHARE / s)
        s_res = s * (1 - covered)
        n_band = max(1, int(round(covered * n_out)))
        d_band = draw(n_band, mu_out)
        hat_mu_out = d_band / n_band
        di_hat = (1 - s) * hat_mu_in + s * covered * hat_mu_out
        eff_res = s_res * (M - 1)
        lo = 1 - (di_hat + eff_res * hat_mu_in) / hat_DI0
        se_band = np.sqrt(np.maximum(hat_mu_out * (1 - hat_mu_out) / n_band, 1e-12) * deff)
        se = np.sqrt(((1 - s) * se_in) ** 2 + (s * covered * se_band) ** 2
                     + (eff_res * se_in) ** 2 + (lo * se_0 / hat_DI0) ** 2) / hat_DI0
    elif design == "full_randomisation":
        d_out = draw(n_out, mu_out)
        hat_mu_out = d_out / n_out
        di_hat = (1 - s) * hat_mu_in + s * hat_mu_out
        lo = 1 - di_hat / hat_DI0
        se_out = np.sqrt(np.maximum(hat_mu_out * (1 - hat_mu_out) / n_out, 1e-12) * deff)
        se = np.sqrt(((1 - s) * se_in) ** 2 + (s * se_out) ** 2
                     + (di_hat * se_0 / hat_DI0) ** 2) / hat_DI0
    else:
        raise ValueError(design)
    return float(np.mean(lo - 1.645 * se > 0.0))


def main():
    t0 = time.time()
    DI0, s, M = load_params()
    rng = np.random.default_rng(0)
    rows = []
    for design in ["observational", "randomised_band", "full_randomisation"]:
        for n in N_GRID:
            for m in MONTH_GRID:
                pw = power_cell(n, m, DI0, s, M, design, rng)
                rows.append(dict(source="simulator", design=design, n=n, months=m, power=pw,
                                 DI0=DI0, swap_share=s, true_M=M, true_reduction=TRUE_R,
                                 band_share=BAND_SHARE, rho_month=RHO_MONTH))
    dump("Q4_power_surfaces.json", rows)

    def smallest(design):
        ok = [r for r in rows if r["design"] == design and r["power"] >= 0.80]
        if not ok:
            return None
        ok.sort(key=lambda r: (r["n"], r["months"]))
        return {"n": ok[0]["n"], "months": ok[0]["months"], "power": ok[0]["power"]}

    summary = dict(source="simulator", DI0=DI0, swap_share=s, true_M=M,
                   base_rate_note="DI0 is the simulator incumbent's inflow at a=0.7",
                   smallest_80={d: smallest(d) for d in
                                ["observational", "randomised_band", "full_randomisation"]},
                   band_share=BAND_SHARE, rho_month=RHO_MONTH, reps=REPS,
                   runtime_s=time.time() - t0)
    o, b = summary["smallest_80"]["observational"], summary["smallest_80"]["randomised_band"]
    if o and b:
        summary["cohort_saving_from_band"] = 1.0 - b["n"] / o["n"]
    dump("Q4_power_summary.json", summary)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
