"""SBA National: Q1 attainability (ceiling band + achieved inflow) and the
per-coalition fits that Q3 reuses.

Sources (grouping fixed in 07_sba_prepare.py): firmographics (F), guarantee (G), bank (B).
Incumbents: L2-logistic and tuned LightGBM on F union {Term, log GrAppv}.
Split: out-of-time by approval fiscal year.

Writes results/SBA_Q1_cells.json, results/SBA_Q1_summary.json, results/SBA_incumbents.json
and data/processed/sba_scores.npz (scores cached for the later SBA scripts).
"""
from __future__ import annotations

import itertools
import os
import time

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import APPROVAL_RATES, A_OP, dump
from src.eval.dir_metric import default_inflow, risk_to_score  # noqa: E402

from common import PROC  # the repository root is resolved once, in common.py
SOURCES = ["firmographics", "guarantee", "bank"]
N_ENS = 2          # two ensemble members, to keep the run time manageable
T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)


def load():
    d = np.load(os.path.join(PROC, "sba.npz"), allow_pickle=True)
    return d


def coalition_cols(cols, src, C):
    return np.array([i for i, s in enumerate(src) if s in C], int)


def ceiling_band(X, y, seed=0, n_folds=3, lgb_kw=None):
    """Three estimators of Section 4.1 on the held-out window, cross-fitted.
    Returns eta_hat (cross-fitted) and the ensemble spread per row."""
    n = len(y)
    eta = np.zeros(n)
    ens = np.zeros((N_ENS, n))
    for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(X, y):
        m = LGBMClassifier(random_state=seed, **lgb_kw).fit(X[tr], y[tr])
        eta[te] = m.predict_proba(X[te])[:, 1]
        rs = np.random.default_rng(1000 + seed)
        for b in range(N_ENS):
            sub = rs.choice(len(tr), int(0.7 * len(tr)), replace=False)
            mb = LGBMClassifier(random_state=seed * 31 + b, **lgb_kw).fit(X[tr][sub], y[tr][sub])
            ens[b, te] = mb.predict_proba(X[te])[:, 1]
    return eta, ens.std(0)


def band_at(eta, spread, y, a):
    """(point, lower, upper) for DI*(S,a) from the three estimators."""
    k = max(1, int(round(a * len(eta))))
    tail = np.argsort(eta)[:k]
    plugin = float(eta[tail].mean())
    realised = float(y[tail].mean())
    point = float(plugin + 0.5 * (realised - plugin))
    lower = float(max(0.0, plugin - float(spread[tail].mean())))
    upper = float(max(realised, point))
    return point, lower, upper, plugin, realised


def main():
    d = load()
    X, y, test = d["X"], d["y"].astype(float), d["test"].astype(bool)
    cols, src, inc_cols = list(d["cols"]), list(d["src"]), list(d["inc_cols"])
    fy = d["fy"]
    tr, te = ~test, test
    log(f"n_train={tr.sum():,} n_test={te.sum():,} base_tr={y[tr].mean():.4f} base_te={y[te].mean():.4f}")

    # ---- incumbents --------------------------------------------------------
    ic = np.array([cols.index(c) for c in inc_cols], int)
    Xi_tr, Xi_te = np.nan_to_num(X[np.ix_(tr, ic)]), np.nan_to_num(X[np.ix_(te, ic)])
    ytr, yte = y[tr], y[te]

    # validation window = last two fiscal years of the training window
    fytr = fy[tr]
    vcut = np.sort(np.unique(fytr))[-2]
    vmask = fytr >= vcut
    log(f"validation window: FY >= {vcut} ({vmask.sum():,} rows)")

    grid = [dict(num_leaves=nl, n_estimators=300, learning_rate=lr, min_child_samples=50)
            for nl in (31, 63) for lr in (0.05, 0.1)]
    rgt = np.random.default_rng(0)
    fit_ix = np.where(~vmask)[0]
    if len(fit_ix) > 150_000:
        fit_ix = rgt.choice(fit_ix, 150_000, replace=False)
    best, best_auc = None, -1.0
    for g in grid:
        m = LGBMClassifier(random_state=0, verbose=-1, n_jobs=2, force_col_wise=True, **g)
        m.fit(Xi_tr[fit_ix], ytr[fit_ix])
        auc = roc_auc_score(ytr[vmask], m.predict_proba(Xi_tr[vmask])[:, 1])
        log(f"  tune {g} val AUC {auc:.4f}")
        if auc > best_auc:
            best, best_auc = g, auc
    log(f"tuned incumbent hyperparameters {best} (validation AUC {best_auc:.4f})")
    LGB = dict(verbose=-1, n_jobs=2, force_col_wise=True, **best)
    # ceiling learner fixed so the band's cost does not scale with the tuned setting
    CEIL_LGB = dict(verbose=-1, n_jobs=2, force_col_wise=True, num_leaves=31,
                    n_estimators=200, learning_rate=0.08, min_child_samples=50)

    m_gbdt = LGBMClassifier(random_state=0, **LGB).fit(Xi_tr, ytr)
    f0_gbdt = risk_to_score(m_gbdt.predict_proba(Xi_te)[:, 1])
    m_log = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    m_log.fit(Xi_tr, ytr)
    f0_log = risk_to_score(m_log.predict_proba(Xi_te)[:, 1])
    inc = dict(source="SBA", tuned_hyperparameters=best, validation_auc=float(best_auc),
               validation_from_fy=int(vcut),
               auc_tuned_gbdt=float(roc_auc_score(yte, -f0_gbdt)),
               auc_l2_logistic=float(roc_auc_score(yte, -f0_log)),
               fields=inc_cols)
    for a in APPROVAL_RATES:
        inc[f"DI0_tuned_a{a}"] = default_inflow(f0_gbdt, yte, a)
        inc[f"DI0_logistic_a{a}"] = default_inflow(f0_log, yte, a)
    dump("SBA_incumbents.json", inc)
    log("incumbents done", {k: round(v, 4) for k, v in inc.items() if k.startswith("auc")})

    # ---- coalitions --------------------------------------------------------
    cells, scores = [], {"f0_gbdt": f0_gbdt, "f0_log": f0_log, "y_test": yte}
    pi = float(ytr.mean())
    subsets = [c for r in range(4) for c in itertools.combinations(SOURCES, r)]
    for C in subsets:
        tag = "+".join(s[0].upper() for s in C) if C else "EMPTY"
        if not C:
            rng = np.random.default_rng(7)
            f = rng.random(te.sum())
            eta = np.full(te.sum(), float(yte.mean()))
            spread = np.zeros(te.sum())
            auc = 0.5
        else:
            ci = coalition_cols(cols, src, set(C))
            Xtr_c, Xte_c = X[np.ix_(tr, ci)], X[np.ix_(te, ci)]
            m = LGBMClassifier(random_state=0, **LGB).fit(Xtr_c, ytr)
            f = risk_to_score(m.predict_proba(Xte_c)[:, 1])
            auc = float(roc_auc_score(yte, -f))
            eta, spread = ceiling_band(Xte_c, yte, seed=0, lgb_kw=CEIL_LGB)
        scores[f"f_{tag}"] = f
        scores[f"eta_{tag}"] = eta
        scores[f"spread_{tag}"] = spread
        for a in APPROVAL_RATES:
            pt, lo, hi, plugin, realised = band_at(eta, spread, yte, a)
            di = default_inflow(f, yte, a)
            di0g = inc[f"DI0_tuned_a{a}"]
            di0l = inc[f"DI0_logistic_a{a}"]
            cells.append(dict(
                source="SBA", coalition=tag, sources=list(C), a=a, auc=auc,
                DI_achieved=di, DI_star_point=pt, DI_star_lo=lo, DI_star_hi=hi,
                DI_star_plugin=plugin, DI_star_realised=realised,
                DI0_tuned=di0g, DI0_logistic=di0l,
                gap_pp=100.0 * (di - pt), gap_lo_pp=100.0 * (di - hi), gap_hi_pp=100.0 * (di - lo),
                DIR_tuned=1.0 - di / di0g, DIR_logistic=1.0 - di / di0l,
                DIRstar_tuned=1.0 - pt / di0g, DIRstar_tuned_lo=1.0 - hi / di0g,
                DIRstar_tuned_hi=1.0 - lo / di0g,
                DIRstar_logistic=1.0 - pt / di0l,
                band_width_pp=100.0 * (hi - lo)))
        dump("SBA_Q1_cells.json", cells)
        log(f"coalition {tag} auc={auc:.4f} done")

    np.savez_compressed(os.path.join(PROC, "sba_scores.npz"), **scores)

    # ---- summary endpoints -------------------------------------------------
    def cell(tag, a):
        return next(c for c in cells if c["coalition"] == tag and c["a"] == a)

    full = "F+G+B"
    c07, c05 = cell(full, 0.7), cell(full, 0.5)
    clears = [c["coalition"] for c in cells
              if c["a"] == A_OP and c["coalition"] != "EMPTY" and c["DIRstar_tuned"] >= 0.15]
    n_cfg = len([s for s in subsets if s])
    # model-or-data reading: the shortfall is a modelling problem where the gap is the larger
    # part of the distance from achieved inflow to the 15% line, a data problem where the
    # ceiling itself sits above the line.
    md = {}
    for c in [x for x in cells if x["a"] == A_OP and x["coalition"] != "EMPTY"]:
        md[c["coalition"]] = "data" if c["DIRstar_tuned"] < 0.15 else (
            "model" if c["DIR_tuned"] < 0.15 else "neither (target met)")
    summ = dict(
        source="SBA", n_test=int(te.sum()), base_rate_test=float(yte.mean()), base_rate_train=pi,
        gap_a07_pp=c07["gap_pp"], gap_a07_band_pp=[c07["gap_lo_pp"], c07["gap_hi_pp"]],
        gap_a05_pp=c05["gap_pp"],
        band_a07=[c07["DI_star_lo"], c07["DI_star_hi"]], band_width_a07_pp=c07["band_width_pp"],
        DI_star_a07=c07["DI_star_point"], DI_achieved_a07=c07["DI_achieved"],
        DI0_tuned_a07=c07["DI0_tuned"], DI0_logistic_a07=c07["DI0_logistic"],
        attainable_reduction_tuned_a07=c07["DIRstar_tuned"],
        attainable_reduction_tuned_a07_band=[c07["DIRstar_tuned_lo"], c07["DIRstar_tuned_hi"]],
        attainable_reduction_logistic_a07=c07["DIRstar_logistic"],
        difference_between_incumbents=c07["DIRstar_tuned"] - c07["DIRstar_logistic"],
        achieved_DIR_tuned_a07=c07["DIR_tuned"], achieved_DIR_logistic_a07=c07["DIR_logistic"],
        configurations_clearing_15=clears, n_configurations=n_cfg,
        model_or_data_by_configuration=md,
    )
    dump("SBA_Q1_summary.json", summ)
    log("SUMMARY", {k: v for k, v in summ.items() if not isinstance(v, (dict, list))})


if __name__ == "__main__":
    main()
