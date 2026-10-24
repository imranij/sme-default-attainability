"""The simulator side of Q2: Assumption 1 and the standard corrections
across the five selection mechanisms of Section 5.1.

For each mechanism the simulator's TRUE reject outcomes give mu_out, hence the true
risk ratio M and the true DIR. We record whether the lower inequality of Assumption 1
holds, whether the true DIR lies in the identified set at the true M, and whether
MAR reweighting, a Heckman-type selection model and fuzzy augmentation land inside
that set.  Every number here is simulator output.
"""
from __future__ import annotations

import time

import numpy as np
from lightgbm import LGBMClassifier
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import dump


def LR():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
from simfeat import build, selection

A0, A_NEW = 0.7, 0.6
LGB_KW = dict(n_estimators=80, learning_rate=0.12, num_leaves=31, min_child_samples=50,
              verbose=-1, n_jobs=2)


def topk(score, a):
    k = max(1, int(round(a * len(score))))
    return np.argsort(-score)[:k]          # score = credit quality, higher = better


def identified_set(mu_in, s, DI0, M):
    hi = 1.0 - mu_in / DI0
    lo = 1.0 - mu_in * (1.0 + s * (M - 1.0)) / DI0
    return lo, hi


def m_star(mu_in, s, DI0, r=0.15):
    return 1.0 + (1.0 / s) * (((1.0 - r) * DI0) / mu_in - 1.0)


def run_mechanism(mech, static, X, y, seed=0):
    appr = selection(static, y, mech, a0=A0, seed=seed)
    Xv = X.to_numpy(float)
    obs = appr == 1

    # new model, trained on the accepted sample only (what a lender can do)
    m = LGBMClassifier(random_state=seed, **LGB_KW).fit(Xv[obs], y[obs])
    risk_new = m.predict_proba(Xv)[:, 1]
    f_new = -risk_new                                  # credit-quality convention
    f_old = -static.incumbent_score.to_numpy()

    A = set(topk(f_new, A_NEW))
    A0set = set(np.flatnonzero(appr))
    inter = np.array(sorted(A & A0set), int)
    outer = np.array(sorted(A - A0set), int)
    s = len(outer) / max(1, len(A))
    mu_in = float(y[inter].mean())
    mu_out = float(y[outer].mean()) if len(outer) else float("nan")
    true_M = mu_out / mu_in if mu_in > 0 else float("nan")

    old_appr = np.array(sorted(set(topk(f_old, A_NEW)) & A0set), int)
    DI0 = float(y[old_appr].mean())
    di_new_true = float(y[np.array(sorted(A), int)].mean())
    true_dir = 1.0 - di_new_true / DI0

    lo, hi = identified_set(mu_in, s, DI0, true_M if np.isfinite(true_M) else 1.0)
    lo1, hi1 = identified_set(mu_in, s, DI0, 1.0)

    # ---- the three standard corrections -------------------------------------
    # (a) MAR reweighting: inverse propensity of acceptance, fitted on observed x.
    pm = LR().fit(Xv, appr)
    prop = np.clip(pm.predict_proba(Xv)[:, 1], 0.02, 0.98)
    w = 1.0 / prop
    def wmean(ix):
        return float(np.average(y[ix], weights=w[ix]))
    ix_new = np.array(sorted(A), int)
    ix_new_obs = ix_new[appr[ix_new] == 1]
    ix_old_obs = old_appr
    dir_mar = 1.0 - wmean(ix_new_obs) / wmean(ix_old_obs)

    # (b) Heckman-type two-step: probit selection, then outcome with the inverse Mills ratio.
    zsel = norm.ppf(np.clip(prop, 1e-4, 1 - 1e-4))
    mills = norm.pdf(zsel) / np.clip(norm.cdf(zsel), 1e-6, 1)
    Zo = np.column_stack([Xv[obs], mills[obs]])
    om = LR().fit(Zo, y[obs])
    Zall = np.column_stack([Xv, mills])
    yhat_h = om.predict_proba(Zall)[:, 1]
    def hmean(ix):
        return float(np.where(appr[ix] == 1, y[ix], yhat_h[ix]).mean())
    dir_heck = 1.0 - hmean(ix_new) / hmean(ix_old_obs)

    # (c) fuzzy augmentation: rejects enter twice, weighted by the model's predicted risk.
    yhat_f = risk_new
    def fmean(ix):
        return float(np.where(appr[ix] == 1, y[ix], yhat_f[ix]).mean())
    dir_fuzzy = 1.0 - fmean(ix_new) / fmean(ix_old_obs)

    def inside(v):
        return bool(lo - 1e-12 <= v <= hi + 1e-12)

    return dict(source="simulator", mechanism=mech, seed=seed, a0=A0, a=A_NEW,
                swap_share=s, mu_in=mu_in, mu_out=mu_out, true_M=true_M, DI0=DI0,
                di_new_true=di_new_true, true_DIR=true_dir,
                set_lo_at_trueM=lo, set_hi=hi, set_lo_at_M1=lo1,
                lower_inequality_holds=bool(mu_out >= mu_in),
                true_DIR_in_set=inside(true_dir),
                M_star_015=m_star(mu_in, s, DI0),
                dir_mar=dir_mar, dir_heckman=dir_heck, dir_fuzzy=dir_fuzzy,
                mar_inside=inside(dir_mar), heckman_inside=inside(dir_heck),
                fuzzy_inside=inside(dir_fuzzy))


def main():
    t0 = time.time()
    static, X, y, _ = build(n_firms=12_000, seed=0)
    print("sim built", len(static), "base rate", y.mean(), f"{time.time()-t0:.0f}s", flush=True)
    rows = []
    for mech in ["score", "score_plus_unobservable", "non_monotone", "human_override", "policy_floor"]:
        for seed in range(1):
            r = run_mechanism(mech, static, X, y, seed=seed)
            rows.append(r)
            print(mech, seed, "M=%.2f" % r["true_M"], "s=%.3f" % r["swap_share"],
                  "inside", r["mar_inside"], r["heckman_inside"], r["fuzzy_inside"],
                  f"{time.time()-t0:.0f}s", flush=True)
    dump("Q2_sim_mechanisms.json", rows)

    Ms = [r["true_M"] for r in rows if np.isfinite(r["true_M"])]
    by = {}
    for r in rows:
        by.setdefault(r["mechanism"], []).append(r)
    summary = dict(
        source="simulator", base_rate=float(y.mean()), n_firms=int(len(static)),
        min_true_M=float(min(Ms)), max_true_M=float(max(Ms)),
        mechanisms_lower_inequality_holds=[m for m, v in by.items()
                                           if all(x["lower_inequality_holds"] for x in v)],
        mechanisms_lower_inequality_fails=[m for m, v in by.items()
                                           if not all(x["lower_inequality_holds"] for x in v)],
        mechanisms_true_dir_in_set=[m for m, v in by.items() if all(x["true_DIR_in_set"] for x in v)],
        mechanisms_mar_fails=[m for m, v in by.items() if not all(x["mar_inside"] for x in v)],
        mechanisms_heckman_fails=[m for m, v in by.items() if not all(x["heckman_inside"] for x in v)],
        mechanisms_fuzzy_fails=[m for m, v in by.items() if not all(x["fuzzy_inside"] for x in v)],
        mean_true_M_by_mechanism={m: float(np.mean([x["true_M"] for x in v])) for m, v in by.items()},
        runtime_s=time.time() - t0)
    summary["mechanisms_any_standard_method_fails"] = sorted(set(
        summary["mechanisms_mar_fails"] + summary["mechanisms_heckman_fails"]
        + summary["mechanisms_fuzzy_fails"]))
    dump("Q2_sim_summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
