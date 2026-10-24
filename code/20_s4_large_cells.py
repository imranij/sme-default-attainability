"""Supplementary S4: the two large ceiling-estimator cells at the (n, base rate)
of SBA National and of Home Credit.

These are the cells `01_ceiling_validation.py` specifies but leaves to a separate step because
of their run time. They are SIMULATOR cells on the controlled DGP of
`common.known_eta_dgp`, not real-data quantities: they are what the paper's sentence
"the band is ... wide at the sample size and base rate of SBA and ... at those of Home Credit"
names. Both (n, base rate) pairs are now known exactly and are the values in `common.py`
(SBA_N = 899,164 / 0.18; HC_N = 307,511 / 0.081), each confirmed against the delivered file.

Identical estimator code to 01_ceiling_validation.estimators; only the grid differs.
Writes results/S4_ceiling_large_cells.json and results/S4_ceiling_summary.json.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from common import (A_OP, HC_BASE, HC_N, RESULTS, SBA_BASE, SBA_N, dump,
                    known_eta_dgp, true_ceiling)

import importlib
_v = importlib.import_module("01_ceiling_validation") if False else None
# 01_... is not a legal module name; re-state the estimator with n_jobs=2 for the large cells.
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

N_ENSEMBLE = 3
LGB_KW = dict(n_estimators=60, learning_rate=0.12, num_leaves=15, min_child_samples=40,
              verbose=-1, n_jobs=2, force_col_wise=True)
T0 = time.time()


def estimators(X, y, a, seed):
    n = len(y)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    eta_hat = np.zeros(n)
    ens = np.zeros((N_ENSEMBLE, n), dtype="float32")
    for tr, te in skf.split(X, y):
        m = LGBMClassifier(random_state=seed, **LGB_KW).fit(X[tr], y[tr])
        eta_hat[te] = m.predict_proba(X[te])[:, 1]
        for b in range(N_ENSEMBLE):
            rs = np.random.default_rng(1000 * seed + b)
            sub = rs.choice(len(tr), int(0.7 * len(tr)), replace=False)
            mb = LGBMClassifier(random_state=seed * 31 + b, **LGB_KW).fit(X[tr][sub], y[tr][sub])
            ens[b, te] = mb.predict_proba(X[te])[:, 1]
        print(f"  fold done t={time.time()-T0:.0f}s", flush=True)
    k = max(1, int(round(a * n)))
    tail = np.argsort(eta_hat)[:k]
    plugin = float(eta_hat[tail].mean())
    realised = float(y[tail].mean())
    point = float(plugin + 0.5 * (realised - plugin))
    spread = float(ens.std(0)[tail].mean())
    lower = float(max(0.0, plugin - spread))
    upper = max(realised, point)
    return point, lower, upper, plugin, realised, spread


def main():
    big = []
    for tag, n, br in [("SBA-like", SBA_N, SBA_BASE), ("HomeCredit-like", HC_N, HC_BASE)]:
        for seed in range(1):
            X, y, eta = known_eta_dgp(n, 12, br, 1.0, seed=seed)
            truth = true_ceiling(eta, A_OP)
            del eta
            pt, lo, hi, plugin, realised, spread = estimators(X, y, A_OP, seed)
            del X, y
            big.append(dict(source="simulator", study="S4_ceiling_large", cell=tag, n=n,
                            base_rate=br, d=12, noise=1.0, seed=seed, a=A_OP,
                            true_ceiling=truth, point=pt, band_lo=lo, band_hi=hi,
                            plugin=plugin, realised=realised, ens_spread=spread,
                            bias_pp=100.0 * (pt - truth),
                            plugin_bias_pp=100.0 * (plugin - truth),
                            covered=bool(lo <= truth <= hi),
                            band_width_pp=100.0 * (hi - lo),
                            band_width_rel_to_ceiling=float((hi - lo) / truth)))
            dump("S4_ceiling_large_cells.json", big)
            print(f"large {tag} width={big[-1]['band_width_pp']:.3f}pp "
                  f"t={time.time()-T0:.0f}s", flush=True)

    # rebuild the S4 summary now that the two large cells exist
    cells = json.load(open(os.path.join(RESULTS, "S4_ceiling_cells.json")))
    bias = np.array([c["bias_pp"] for c in cells])
    widths = np.array([c["band_width_pp"] for c in cells])
    groups = {}
    for c in cells:
        groups.setdefault((c["n"], c["base_rate"]), []).append(c["bias_pp"])
    summary = dict(
        source="simulator", n_cells=len(cells),
        mean_bias_pp=float(bias.mean()), median_bias_pp=float(np.median(bias)),
        mean_plugin_bias_pp=float(np.mean([c["plugin_bias_pp"] for c in cells])),
        band_coverage=float(np.mean([c["covered"] for c in cells])), nominal=0.95,
        mean_band_width_pp=float(widths.mean()),
        frac_uninformative=float(np.mean([c["band_width_pp"] > 0.15 * 100.0 * c["true_ceiling"]
                                          for c in cells])),
        n_groups=len(groups),
        n_groups_with_sign_change=int(sum(1 for v in groups.values() if min(v) < 0 < max(v))),
        band_width_sba_like_pp=float(np.mean([b["band_width_pp"] for b in big
                                              if b["cell"] == "SBA-like"])),
        band_width_hc_like_pp=float(np.mean([b["band_width_pp"] for b in big
                                             if b["cell"] == "HomeCredit-like"])),
        large_cells_note=("simulator cells on the controlled DGP at the (n, base rate) of the two "
                          "real datasets; not real-data band widths"),
        runtime_s=time.time() - T0)
    dump("S4_ceiling_summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
