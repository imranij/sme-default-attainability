"""SBA National: the component analyses and the fairness tables (component ablation in Section 7 and the Supplementary Material).

  * minimum detectable difference in DIR at a = 0.7 (Supplementary S14)
  * the two objectives of Section 3.4 and the kappa / lambda sweep
  * the omitted implicit-function term's share of the gradient's L1 mass
  * AUC vs DIR ordering across the approval-rate grid, and the pairs that reverse
  * reason-code stability (rank-biased overlap) across bootstrap and out-of-time refits
  * margin-based abstention gain per unit of the recovery fraction rho
  * approval-rate and default-rate gaps by sector, region and size band
"""
from __future__ import annotations

import itertools
import os
import time

import lightgbm as lgb
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

from common import APPROVAL_RATES, A_OP, dump
from src.eval.dir_metric import _approved_index, default_inflow, dir_metric, risk_to_score  # noqa
from src.models.dial_objective import (lightgbm_dial_objective, surrogate_terms,  # noqa
                                       tau_gradient_correction)

from common import PROC  # the repository root is resolved once, in common.py

# --- fairness group labels ---------------------------------------------------
# 07_sba_prepare.py encodes the three fairness dimensions as ordinal codes and does not
# persist the level lists, so the fairness tables named their levels by integer alone.
# The encoding is ordcode(): sorted(unique(level)) over the TRAIN window, level i -> code i,
# unseen test levels -> -1. The lists below are that sorted order, recovered from
# data/raw/sba/SBAnational.csv by replaying the label filter, the maturity filter and
# the fiscal-year split of 07_sba_prepare.py (train window FY < 2006, n_train = 522,254 and
# n_test = 161,244, matching results/SBA_cohort.json). They are applied for display only:
# every rate, share and gap below is computed from the codes and is unchanged by labelling.
# The guard on length means that if the prep is ever re-run on a different extract, and the
# number of levels moves, the tables fall back to codes rather than mislabel them.
NAICS2_SECTOR = {
    "11": "Agriculture, forestry, fishing and hunting",
    "21": "Mining, quarrying, oil and gas extraction",
    "22": "Utilities", "23": "Construction",
    "31": "Manufacturing", "32": "Manufacturing", "33": "Manufacturing",
    "42": "Wholesale trade", "44": "Retail trade", "45": "Retail trade",
    "48": "Transportation and warehousing", "49": "Transportation and warehousing",
    "51": "Information", "52": "Finance and insurance",
    "53": "Real estate, rental and leasing",
    "54": "Professional, scientific and technical services",
    "55": "Management of companies and enterprises",
    "56": "Administrative, support and waste management services",
    "61": "Educational services", "62": "Health care and social assistance",
    "71": "Arts, entertainment and recreation",
    "72": "Accommodation and food services",
    "81": "Other services (except public administration)",
    "92": "Public administration",
}
SECTOR_LEVELS = ["11", "21", "22", "23", "31", "32", "33", "42", "44", "45", "48", "49",
                 "51", "52", "53", "54", "55", "56", "61", "62", "71", "72", "81", "92"]
REGION_LEVELS = ["AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI",
                 "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN",
                 "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH",
                 "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA",
                 "WI", "WV", "WY"]
# np.digitize(NoEmp, [1, 5, 20, 100]) in 07_sba_prepare.py; see acct["size_band_edges"].
SIZE_LEVELS = ["0 employees", "1-4 employees", "5-19 employees",
               "20-99 employees", "100+ employees"]
LEVEL_LABELS = {
    "sector": (SECTOR_LEVELS,
               lambda v: f"{v} {NAICS2_SECTOR[v]}",
               "two-digit NAICS sector of the borrower"),
    "region": (REGION_LEVELS, lambda v: v, "borrower state (USPS abbreviation)"),
    "size_band": (SIZE_LEVELS, lambda v: v, "employee-count band from NoEmp"),
}
LEVEL_NOTE_FALLBACK = (
    "Levels are the ordinal codes of the feature matrix, which are the dataset's own "
    "categories; see 07_sba_prepare.py for the encoding.")


def label_levels(name, codes_present, n_levels_in_data):
    """code -> display label for one fairness dimension, or None if not recoverable."""
    levels, fmt, _ = LEVEL_LABELS[name]
    if n_levels_in_data != len(levels) or max(codes_present, default=-1) >= len(levels):
        return None
    return {c: (fmt(levels[c]) if c >= 0 else "unrecorded (unseen in the train window)")
            for c in codes_present}


SUB_N = 80_000     # training subsample, chosen to keep the run time manageable
SWEEP_N = 30_000
SWEEP_ROUNDS = 60
FINAL_ROUNDS = 100
T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)


def rbo(l1, l2, p=0.9):
    """Rank-biased overlap of two ranked lists (Webber et al. 2010), truncated at len(l1)."""
    s1, s2, k = [], set(), min(len(l1), len(l2))
    A, ov = 0.0, 0
    seen1, seen2 = set(), set()
    for d in range(1, k + 1):
        seen1.add(l1[d - 1]); seen2.add(l2[d - 1])
        ov = len(seen1 & seen2)
        A += (p ** (d - 1)) * ov / d
    return (1 - p) * A + (ov / k) * (p ** k)


def fit_dial(Xtr, ytr, leaves=31, rounds=FINAL_ROUNDS, kappa=0.1, lam=0.5, seed=0):
    ds = lgb.Dataset(Xtr, label=ytr, free_raw_data=False)
    params = dict(num_leaves=leaves, learning_rate=0.08, min_child_samples=50, verbose=-1,
                  num_threads=2, seed=seed, force_col_wise=True,
                  objective=lightgbm_dial_objective(A_OP, kappa, lam))
    return lgb.train(params, ds, num_boost_round=rounds)


def main():
    d = np.load(os.path.join(PROC, "sba.npz"), allow_pickle=True)
    X, y, test = np.nan_to_num(d["X"]), d["y"].astype(float), d["test"].astype(bool)
    cols = [str(c) for c in d["cols"]]
    fy = d["fy"]
    g_sector, g_region, g_size = d["g_sector"], d["g_region"], d["g_size"]
    sc = np.load(os.path.join(PROC, "sba_scores.npz"))
    f0, yte = sc["f0_gbdt"], sc["y_test"].astype(float)
    f_full = sc["f_F+G+B"]
    Xtr, Xte = X[~test], X[test]
    ytr = y[~test]
    rng = np.random.default_rng(0)
    sub = rng.choice(len(ytr), min(SUB_N, len(ytr)), replace=False)

    out = {"source": "SBA"}

    # ---- 1. kappa / lambda sweep and the two objectives --------------------
    fytr = fy[~test]
    vcut = np.sort(np.unique(fytr))[-2]
    vm = fytr >= vcut
    fit_ix = sub[~vm[sub]][:SWEEP_N]
    Xf, yf = Xtr[fit_ix], ytr[fit_ix]
    vsel = np.where(vm)[0]
    vsel = vsel[:40_000]
    Xv, yv = Xtr[vsel], ytr[vsel]
    m0v = LGBMClassifier(num_leaves=31, n_estimators=300, learning_rate=0.08, min_child_samples=50,
                         random_state=0, verbose=-1, n_jobs=2).fit(
        Xf[:, [cols.index(c) for c in d["inc_cols"]]], yf)
    f0v = risk_to_score(m0v.predict_proba(Xv[:, [cols.index(c) for c in d["inc_cols"]]])[:, 1])
    sweep = []
    for kappa in [0.05, 0.1, 0.25, 0.5]:
        for lam in [0.0, 0.1, 0.5, 1.0]:
            b = fit_dial(Xf, yf, kappa=kappa, lam=lam, rounds=SWEEP_ROUNDS)
            fv = risk_to_score(b.predict(Xv, raw_score=True))
            sweep.append(dict(kappa=kappa, lam=lam,
                              DIR_val_a07=float(dir_metric(fv, f0v, yv, A_OP, n_tie_draws=1))))
            log("sweep", kappa, lam, round(sweep[-1]["DIR_val_a07"], 4))
    dump("SBA_kappa_lambda_sweep.json", sweep)
    best = max(sweep, key=lambda r: r["DIR_val_a07"])
    out["best_kappa"], out["best_lambda"] = best["kappa"], best["lam"]
    out["best_kappa_lambda_val_DIR"] = best["DIR_val_a07"]

    # final DIAL fit on the full training subsample at the selected setting
    b_dial = fit_dial(Xtr[sub], ytr[sub], kappa=best["kappa"], lam=best["lam"])
    f_dial = risk_to_score(b_dial.predict(Xte, raw_score=True))
    m_x = LGBMClassifier(num_leaves=31, n_estimators=150, learning_rate=0.08, min_child_samples=50,
                         random_state=0, verbose=-1, n_jobs=2).fit(Xtr[sub], ytr[sub])
    f_xent = risk_to_score(m_x.predict_proba(Xte)[:, 1])
    dir_x = float(dir_metric(f_xent, f0, yte, A_OP))
    dir_d = float(dir_metric(f_dial, f0, yte, A_OP))
    out.update(DIR_xent_a07=dir_x, DIR_dial_a07=dir_d,
               difference_between_objectives=dir_x - dir_d,
               difference_between_objectives_pp=100.0 * (dir_x - dir_d))
    log("objectives", dir_x, dir_d)

    # ---- 2. minimum detectable difference (S6) -----------------------------
    # null variance of the DIR difference: paired bootstrap of the two scores' DIR gap
    nb, n = 250, len(yte)
    rg = np.random.default_rng(1)
    gaps = np.empty(nb)
    for i in range(nb):
        ix = rg.integers(0, n, n)
        gaps[i] = (dir_metric(f_xent[ix], f0[ix], yte[ix], A_OP, n_tie_draws=1, seed=i)
                   - dir_metric(f_dial[ix], f0[ix], yte[ix], A_OP, n_tie_draws=1, seed=i))
    sd = float(gaps.std(ddof=1))
    mdd = float((1.959964 + 0.8416212) * sd)
    out.update(mdd_sd_of_DIR_difference=sd, minimum_detectable_difference=mdd,
               mdd_pp=100.0 * mdd,
               powered=bool(abs(dir_x - dir_d) >= mdd))
    log("MDD", mdd, "observed gap", dir_x - dir_d)

    # ---- 3. omitted implicit-function term ---------------------------------
    fr = b_dial.predict(Xtr[sub][:40_000], raw_score=True)
    yr = ytr[sub][:40_000]
    _, _, _, grad, _ = surrogate_terms(fr, yr, A_OP, best["kappa"], 1.0)
    corr = tau_gradient_correction(fr, yr, A_OP, best["kappa"], 1.0)
    out["omitted_gradient_mass"] = float(np.abs(corr).sum() / max(np.abs(grad).sum(), 1e-12))
    log("omitted gradient mass", out["omitted_gradient_mass"])

    # ---- 4. AUC vs DIR ordering across the approval-rate grid --------------
    cfgs = {}
    for tag in ["F", "G", "B", "F+G", "F+B", "G+B", "F+G+B"]:
        if f"f_{tag}" in sc:
            cfgs[tag] = sc[f"f_{tag}"]
    cfgs["dial"] = f_dial
    aucs = {k: float(roc_auc_score(yte, -v)) for k, v in cfgs.items()}
    dirs = {a: {k: float(dir_metric(v, f0, yte, a, n_tie_draws=1)) for k, v in cfgs.items()}
            for a in APPROVAL_RATES}
    rev, tot = 0, 0
    rev_detail = []
    for a in APPROVAL_RATES:
        for k1, k2 in itertools.combinations(cfgs, 2):
            tot += 1
            if np.sign(aucs[k1] - aucs[k2]) != np.sign(dirs[a][k1] - dirs[a][k2]):
                rev += 1
                rev_detail.append(dict(a=a, pair=[k1, k2]))
    out.update(auc_by_config=aucs, dir_by_config=dirs, reversing_pairs=rev,
               total_pairs=tot, reversing_pairs_detail=rev_detail[:40],
               auc_dir_consistent=bool(rev == 0))
    dump("SBA_auc_dir.json", dict(auc=aucs, dir=dirs, reversing=rev, total=tot,
                                  detail=rev_detail, source="SBA"))
    log("AUC-DIR reversals", rev, "of", tot)

    # ---- 5. reason-code stability (rank-biased overlap) --------------------
    ns = 1500
    si = rng.choice(len(yte), ns, replace=False)
    Xs = Xte[si]

    def shap_ranks(model, native=False):
        c = (model.predict(Xs, pred_contrib=True) if native
             else model.booster_.predict(Xs, pred_contrib=True))
        c = np.asarray(c)[:, :-1]
        order = np.argsort(-np.abs(c), axis=1)
        return order

    base = LGBMClassifier(num_leaves=31, n_estimators=150, learning_rate=0.08,
                          min_child_samples=50, random_state=0, verbose=-1, n_jobs=2).fit(
        Xtr[sub], ytr[sub])
    r_base = shap_ranks(base)
    boot_r = []
    for s in range(1, 3):
        rs = np.random.default_rng(100 + s)
        bi = rs.integers(0, len(sub), len(sub))
        m = LGBMClassifier(num_leaves=31, n_estimators=150, learning_rate=0.08,
                           min_child_samples=50, random_state=s, verbose=-1, n_jobs=2).fit(
            Xtr[sub][bi], ytr[sub][bi])
        boot_r.append(shap_ranks(m))
    yrs = np.sort(np.unique(fytr))
    half = yrs[len(yrs) // 2]
    oot_r = []
    for msk in [fytr < half, fytr >= half]:
        idx = np.where(msk)[0]
        idx = rng.choice(idx, min(len(idx), SUB_N), replace=False)
        m = LGBMClassifier(num_leaves=31, n_estimators=150, learning_rate=0.08,
                           min_child_samples=50, random_state=0, verbose=-1, n_jobs=2).fit(
            Xtr[idx], ytr[idx])
        oot_r.append(shap_ranks(m))
    r_dial = shap_ranks(b_dial, native=True)

    def mean_rbo(A, B):
        return float(np.mean([rbo(list(A[i]), list(B[i])) for i in range(len(A))]))

    out["rbo_bootstrap"] = float(np.mean([mean_rbo(r_base, r) for r in boot_r]))
    out["rbo_out_of_time"] = mean_rbo(oot_r[0], oot_r[1])
    out["rbo_dial_vs_xent"] = mean_rbo(r_base, r_dial)
    out["rbo_stability_shift"] = out["rbo_dial_vs_xent"] - out["rbo_bootstrap"]
    log("RBO", out["rbo_bootstrap"], out["rbo_out_of_time"], out["rbo_dial_vs_xent"])

    # ---- 6. margin-based abstention ----------------------------------------
    a = A_OP
    idx_ap = _approved_index(f_full, a, np.random.default_rng(0))
    tau = float(np.sort(f_full)[::-1][len(idx_ap) - 1])
    margin = np.abs(f_full - tau)
    ab = {}
    for b_budget in [0.02, 0.05, 0.10]:
        k = int(round(b_budget * len(f_full)))
        routed = np.argsort(margin)[:k]
        routed_ap = np.intersect1d(routed, idx_ap)
        di = float(yte[idx_ap].mean())
        # per unit of the recovery fraction rho: rho * (defaults routed out) / (approvals)
        per_rho = float(yte[routed_ap].sum() / len(idx_ap))
        ab[f"b={b_budget}"] = dict(n_routed=int(k), n_routed_approved=int(len(routed_ap)),
                                   DI_a07=di, gain_pp_per_unit_rho=100.0 * per_rho,
                                   DIR_gain_per_unit_rho=per_rho / di)
    out["abstention"] = ab
    out["selective_prediction_gain_pp_per_rho"] = ab["b=0.05"]["gain_pp_per_unit_rho"]
    log("abstention", ab["b=0.05"])

    # ---- 7. fairness tables -------------------------------------------------
    gt, gr_, gs = g_sector[test], g_region[test], g_size[test]
    fair = {}
    routed05 = np.argsort(margin)[:int(0.05 * len(f_full))]
    ap_mask = np.zeros(len(yte), bool); ap_mask[idx_ap] = True
    ro_mask = np.zeros(len(yte), bool); ro_mask[routed05] = True
    over = []
    for name, g in [("sector", gt), ("region", gr_), ("size_band", gs)]:
        rows = []
        for lev in np.unique(g):
            m = g == lev
            if m.sum() < 500:
                continue
            ar = float(ap_mask[m].mean())
            dr = float(yte[m & ap_mask].mean()) if (m & ap_mask).sum() > 0 else float("nan")
            rr = float(ro_mask[m].mean())
            rows.append(dict(level=int(lev), n=int(m.sum()), approval_rate=ar,
                             default_rate_among_approvals=dr, routed_share=rr,
                             pool_share=float(m.mean()),
                             routed_share_of_routed=float(ro_mask[m].sum() / max(ro_mask.sum(), 1))))
        # display labels for the ordinal codes; numbers above are untouched by this
        n_in_data = int(sum(1 for lev in np.unique(g) if lev >= 0))
        lab = label_levels(name, [r["level"] for r in rows], n_in_data)
        for r in rows:
            r["label"] = lab[r["level"]] if lab else str(r["level"])
        note = (f"Levels are the {LEVEL_LABELS[name][2]}, recovered from the ordinal "
                "encoding of 07_sba_prepare.py." if lab else LEVEL_NOTE_FALLBACK)
        ars = [r["approval_rate"] for r in rows]
        drs = [r["default_rate_among_approvals"] for r in rows if np.isfinite(r["default_rate_among_approvals"])]
        fair[name] = dict(rows=rows, n_levels=len(rows), labelled=bool(lab), note=note,
                          approval_rate_gap_pp=100.0 * (max(ars) - min(ars)),
                          default_rate_gap_pp=100.0 * (max(drs) - min(drs)))
        for r in rows:
            if r["pool_share"] > 0:
                over.append((name, r["level"], r["routed_share_of_routed"] / r["pool_share"],
                             r["pool_share"], r["label"]))
    over.sort(key=lambda t: -t[2])
    out["fairness"] = fair
    out["approval_rate_gaps_pp"] = {k: v["approval_rate_gap_pp"] for k, v in fair.items()}
    out["default_rate_gaps_pp"] = {k: v["default_rate_gap_pp"] for k, v in fair.items()}
    out["routed_over_representation_top"] = [
        dict(dimension=t[0], level=int(t[1]), ratio=float(t[2]), pool_share=float(t[3]),
             label=t[4])
        for t in over[:6]]
    dump("SBA_fairness.json", fair)
    dump("SBA_components.json", out)
    log("done")


if __name__ == "__main__":
    main()
