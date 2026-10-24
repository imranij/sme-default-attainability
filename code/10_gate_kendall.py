"""The sim-to-real gate, SBA arm only (Section 5.2).

The pre-specified rule needs the Kendall tau between the simulator's ranking of the
learner configurations and their rankings on BOTH SBA and Home Credit. Only the SBA arm
is computed here, so the gate is left UNDECIDED in this step; 19_gate_hc.py adds the Home
Credit arm and takes the decision.

Learner configurations (fixed here): the cross product of the two
objectives of Section 3.4 (cross-entropy; the rate-constrained DIAL surrogate) with three
learner capacities (num_leaves 15 / 31 / 63), six configurations, ranked by DIR at
a = 0.7 against the tuned GBDT incumbent under the same protocol on each side.
"""
from __future__ import annotations

import os
import time

import lightgbm as lgb
import numpy as np
from lightgbm import LGBMClassifier
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score

from common import A_OP, dump
from src.eval.dir_metric import default_inflow, dir_metric, risk_to_score  # noqa: E402
from src.models.dial_objective import lightgbm_dial_objective  # noqa: E402

from common import PROC  # the repository root is resolved once, in common.py
LEAVES = [15, 31, 63]
OBJECTIVES = ["xent", "dial"]
SUB_N = 40_000           # common training subsample; DIAL is O(30n) per boosting round
KAPPA, LAM = 0.1, 0.5
T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)


def fit_xent(Xtr, ytr, Xte, leaves, seed=0, rounds=80):
    m = LGBMClassifier(num_leaves=leaves, n_estimators=rounds, learning_rate=0.08,
                       min_child_samples=50, random_state=seed, verbose=-1, n_jobs=2,
                       force_col_wise=True).fit(Xtr, ytr)
    return risk_to_score(m.predict_proba(Xte)[:, 1])


def fit_dial(Xtr, ytr, Xte, leaves, seed=0, rounds=80):
    """LightGBM 4.x: the custom objective is passed through params['objective']."""
    ds = lgb.Dataset(Xtr, label=ytr, free_raw_data=False)
    params = dict(num_leaves=leaves, learning_rate=0.08, min_child_samples=50, verbose=-1,
                  num_threads=2, seed=seed, force_col_wise=True,
                  objective=lightgbm_dial_objective(A_OP, KAPPA, LAM))
    bst = lgb.train(params, ds, num_boost_round=rounds)
    return risk_to_score(bst.predict(Xte, raw_score=True))


def rank_configs(Xtr, ytr, Xte, yte, f0, tag, seed=0):
    out = []
    rng = np.random.default_rng(seed)
    if len(ytr) > SUB_N:
        ix = rng.choice(len(ytr), SUB_N, replace=False)
        Xtr, ytr = Xtr[ix], ytr[ix]
    for obj in OBJECTIVES:
        for lv in LEAVES:
            t = time.time()
            f = (fit_dial(Xtr, ytr, Xte, lv, seed) if obj == "dial"
                 else fit_xent(Xtr, ytr, Xte, lv, seed))
            d = dir_metric(f, f0, yte, A_OP)
            out.append(dict(dataset=tag, objective=obj, num_leaves=lv, DIR_a07=float(d),
                            DI_a07=default_inflow(f, yte, A_OP),
                            auc=float(roc_auc_score(yte, -f)), fit_s=round(time.time() - t, 1)))
            log(f"{tag} {obj} leaves={lv} DIR={d:.4f} ({out[-1]['fit_s']}s)")
    return out


def main():
    rows = []
    # ---------------- SBA arm ----------------
    d = np.load(os.path.join(PROC, "sba.npz"), allow_pickle=True)
    X, y, test = d["X"], d["y"].astype(float), d["test"].astype(bool)
    sc = np.load(os.path.join(PROC, "sba_scores.npz"))
    f0 = sc["f0_gbdt"]
    rows += rank_configs(np.nan_to_num(X[~test]), y[~test], np.nan_to_num(X[test]), y[test],
                         f0, "SBA")
    dump("gate_configs.json", rows)

    # ---------------- simulator arm ----------------
    from simfeat import build
    static, Xs, ys, frames = build(n_firms=15_000, months=24, seed=0)
    Xs = Xs.to_numpy(dtype="float32")
    ntr = int(0.7 * len(ys))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(ys))
    tr_i, te_i = perm[:ntr], perm[ntr:]
    # simulator incumbent: tuned GBDT on the "static" sources (ratios + firmographics), as on SBA
    static_cols = list(frames["ratios"].columns) + list(frames["firmographics"].columns)
    allc = [c for s in ["ratios", "firmographics", "transactions", "balance", "einvoicing",
                        "engagement", "calendar"] for c in frames[s].columns]
    ic = [allc.index(c) for c in static_cols]
    m0 = LGBMClassifier(num_leaves=31, n_estimators=300, learning_rate=0.05, min_child_samples=50,
                        random_state=0, verbose=-1, n_jobs=2).fit(Xs[np.ix_(tr_i, ic)], ys[tr_i])
    f0s = risk_to_score(m0.predict_proba(Xs[np.ix_(te_i, ic)])[:, 1])
    rows += rank_configs(Xs[tr_i], ys[tr_i], Xs[te_i], ys[te_i], f0s, "simulator")
    dump("gate_configs.json", rows)

    # ---------------- tau ----------------
    key = lambda r: (r["objective"], r["num_leaves"])
    sim = {key(r): r["DIR_a07"] for r in rows if r["dataset"] == "simulator"}
    sba = {key(r): r["DIR_a07"] for r in rows if r["dataset"] == "SBA"}
    ks = sorted(sim)
    tau, p = kendalltau([sim[k] for k in ks], [sba[k] for k in ks])
    # sign agreement among the top three simulator configurations
    top3 = sorted(ks, key=lambda k: -sim[k])[:3]
    pairs = [(a, b) for i, a in enumerate(top3) for b in top3[i + 1:]]
    agree = sum(1 for a, b in pairs if np.sign(sim[a] - sim[b]) == np.sign(sba[a] - sba[b]))
    out = dict(source="SBA+simulator", n_configurations=len(ks),
               configurations=[f"{o} leaves={l}" for o, l in ks],
               kendall_tau_SBA=float(tau), kendall_tau_p=float(p), threshold=0.6,
               kendall_tau_HomeCredit=None,
               top3_simulator=[f"{o} leaves={l}" for o, l in top3],
               top3_sign_agreements_SBA=f"{agree} of {len(pairs)}",
               sba_arm_passes=bool(tau >= 0.6 and agree == len(pairs)),
               gate_decision="undecided",
               reason=("the pre-specified rule of Section 5.2 requires tau on BOTH SBA and "
                       "Home Credit; this file records the SBA arm only, and 19_gate_hc.py adds "
                       "the Home Credit arm and decides the gate"))
    dump("gate_status_sba_arm.json", out)
    log("GATE", out)


if __name__ == "__main__":
    main()
