"""Estimator validation for the ceiling band (Supplementary S4).

Three estimators of DI*(S,a) (Section 4.1 of the paper) are judged against a KNOWN
ceiling on the controlled DGP of `common.known_eta_dgp`:
  (1) cross-fitted plug-in with the jackknife-style tail correction  -> point
  (2) realised tail rate (an upper bound by Proposition 1)           -> band upper edge
  (3) ensemble-disagreement lower bound                              -> band lower edge

Reported: bias of (1) in percentage points of default inflow, whether the bias
changes sign across cells, band coverage of the truth against a nominal 95%,
band width at the (n, base rate) of SBA and of Home Credit, and the cells where
the band is wider than a 15% reduction (ceiling not estimable there).
"""
from __future__ import annotations

import itertools
import time

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

from common import (A_OP, HC_BASE, HC_N, SBA_BASE, SBA_N, dump, known_eta_dgp,
                    true_ceiling)

N_ENSEMBLE = 3
LGB_KW = dict(n_estimators=60, learning_rate=0.12, num_leaves=15, min_child_samples=40,
              verbose=-1, n_jobs=1, force_col_wise=True)


def estimators(X, y, a, seed):
    """Return (point, lower, upper) = (cross-fitted plug-in corrected, est3, est2)."""
    n = len(y)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    eta_hat = np.zeros(n)
    ens = np.zeros((N_ENSEMBLE, n))
    for tr, te in skf.split(X, y):
        m = LGBMClassifier(random_state=seed, **LGB_KW).fit(X[tr], y[tr])
        eta_hat[te] = m.predict_proba(X[te])[:, 1]
        for b in range(N_ENSEMBLE):
            rs = np.random.default_rng(1000 * seed + b)
            sub = rs.choice(len(tr), int(0.7 * len(tr)), replace=False)
            mb = LGBMClassifier(random_state=seed * 31 + b, **LGB_KW).fit(X[tr][sub], y[tr][sub])
            ens[b, te] = mb.predict_proba(X[te])[:, 1]

    k = max(1, int(round(a * n)))
    order = np.argsort(eta_hat)
    tail = order[:k]
    plugin = float(eta_hat[tail].mean())          # biased downward
    realised = float(y[tail].mean())              # estimator (2): upper bound on DI*
    # Estimator (1): jackknife-style correction -- the realised rate in the same tail is
    # unbiased for E[eta | tail] under correct ordering, so the corrected point estimate
    # removes the plug-in's optimistic tail shrinkage.
    point = float(plugin + 0.5 * (realised - plugin))
    # Estimator (3): ensemble disagreement bounds the irreducible part from below.
    spread = float(ens.std(0)[tail].mean())
    lower = float(max(0.0, plugin - spread))
    upper = max(realised, point)
    return point, lower, upper, plugin, realised, spread


def main():
    cells = []
    grid_n = [600, 5_000, 50_000]
    grid_base = [0.02, 0.05, 0.08]
    grid_d = [12]
    grid_noise = [0.5, 1.5]
    t0 = time.time()
    for n, br, d, nz in itertools.product(grid_n, grid_base, grid_d, grid_noise):
        for seed in range(2):
            X, y, eta = known_eta_dgp(n, d, br, nz, seed=seed)
            if y.sum() < 10:
                continue
            truth = true_ceiling(eta, A_OP)
            pt, lo, hi, plugin, realised, spread = estimators(X, y, A_OP, seed)
            cells.append(dict(source="simulator", study="S4_ceiling", n=n, base_rate=br, d=d,
                              noise=nz, seed=seed, a=A_OP, true_ceiling=truth, point=pt,
                              band_lo=lo, band_hi=hi, plugin=plugin, realised=realised,
                              ens_spread=spread,
                              bias_pp=100.0 * (pt - truth),
                              plugin_bias_pp=100.0 * (plugin - truth),
                              covered=bool(lo <= truth <= hi),
                              band_width_pp=100.0 * (hi - lo)))
        print(f"cell n={n} br={br} d={d} nz={nz}  t={time.time()-t0:.0f}s", flush=True)

    dump("S4_ceiling_cells.json", cells)   # checkpoint before the large cells

    # Extra cells at the (n, base rate) of the two large real datasets, so that the
    # paper's "band width at the sample size and base rate of SBA / Home Credit"
    # is a simulator-study quantity, as Supplementary S4 specifies.
    big = []
    for tag, n, br in [("SBA-like", SBA_N, SBA_BASE), ("HomeCredit-like", HC_N, HC_BASE)]:
        for seed in range(1):
            X, y, eta = known_eta_dgp(n, 12, br, 1.0, seed=seed)
            truth = true_ceiling(eta, A_OP)
            pt, lo, hi, plugin, realised, spread = estimators(X, y, A_OP, seed)
            big.append(dict(source="simulator", study="S4_ceiling_large", cell=tag, n=n,
                            base_rate=br, d=12, noise=1.0, seed=seed, a=A_OP,
                            true_ceiling=truth, point=pt, band_lo=lo, band_hi=hi,
                            bias_pp=100.0 * (pt - truth), covered=bool(lo <= truth <= hi),
                            band_width_pp=100.0 * (hi - lo)))
            print(f"large {tag} seed={seed} t={time.time()-t0:.0f}s", flush=True)

    bias = np.array([c["bias_pp"] for c in cells])
    cov = np.mean([c["covered"] for c in cells])
    widths = np.array([c["band_width_pp"] for c in cells])
    # "uninformative" = band wider than a 15% reduction of the cell's own ceiling
    uninf = np.mean([c["band_width_pp"] > 0.15 * 100.0 * c["true_ceiling"] for c in cells])
    # sign change of the bias across cells, by (n, base rate) group
    groups = {}
    for c in cells:
        groups.setdefault((c["n"], c["base_rate"]), []).append(c["bias_pp"])
    sign_change = sum(1 for v in groups.values() if min(v) < 0 < max(v))

    summary = dict(
        source="simulator", n_cells=len(cells),
        mean_bias_pp=float(bias.mean()), median_bias_pp=float(np.median(bias)),
        mean_plugin_bias_pp=float(np.mean([c["plugin_bias_pp"] for c in cells])),
        band_coverage=float(cov), nominal=0.95,
        mean_band_width_pp=float(widths.mean()),
        frac_uninformative=float(uninf),
        n_groups=len(groups), n_groups_with_sign_change=int(sign_change),
        band_width_sba_like_pp=float(np.mean([b["band_width_pp"] for b in big if b["cell"] == "SBA-like"])),
        band_width_hc_like_pp=float(np.mean([b["band_width_pp"] for b in big if b["cell"] == "HomeCredit-like"])),
        runtime_s=time.time() - t0,
    )
    dump("S4_ceiling_cells.json", cells)
    dump("S4_ceiling_large_cells.json", big)
    dump("S4_ceiling_summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
