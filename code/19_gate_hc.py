"""THE SIM-TO-REAL GATE, both arms, and the pre-specified decision (Section 5.2).

The rule of Section 5.2 of the paper: if the Kendall tau between the simulator ranking and
each real-data ranking is below 0.6, OR if the sign of any pairwise difference among the top
three configurations disagrees, no simulator-based ranking is reported and the source-level
analysis is confined to the real datasets.

so the gate PASSES only if tau >= 0.6 on BOTH SBA and Home Credit AND every pairwise
sign among the top three simulator configurations is reproduced on BOTH real datasets.

The six learner configurations are the ones fixed in 10_gate_kendall.py and used for the SBA arm; the
simulator arm is NOT refitted -- the rows already in results/gate_configs.json are reused
verbatim, so the two real arms are compared against one and the same simulator ranking.

Writes results/gate_configs.json (extended with the Home Credit rows) and
results/gate_status.json, which supersedes results/gate_status_sba_arm.json as the single
authoritative record.
"""
from __future__ import annotations

import json
import os
import time

import lightgbm as lgb
import numpy as np
from lightgbm import LGBMClassifier
from scipy.stats import kendalltau
from sklearn.metrics import roc_auc_score

from common import A_OP, RESULTS, dump
from src.eval.dir_metric import default_inflow, dir_metric, risk_to_score  # noqa: E402
from src.models.dial_objective import lightgbm_dial_objective  # noqa: E402

from common import PROC  # the repository root is resolved once, in common.py
LEAVES = [15, 31, 63]
OBJECTIVES = ["xent", "dial"]
SUB_N = 40_000           # the same common training subsample as the SBA arm
KAPPA, LAM = 0.1, 0.5
THRESH = 0.6
T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)


def fit_xent(Xtr, ytr, Xte, leaves, seed=0, rounds=80):
    m = LGBMClassifier(num_leaves=leaves, n_estimators=rounds, learning_rate=0.08,
                       min_child_samples=50, random_state=seed, verbose=-1, n_jobs=2,
                       force_col_wise=True).fit(Xtr, ytr)
    return risk_to_score(m.predict_proba(Xte)[:, 1])


def fit_dial(Xtr, ytr, Xte, leaves, seed=0, rounds=80):
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
    rows = json.load(open(os.path.join(RESULTS, "gate_configs.json")))
    rows = [r for r in rows if r["dataset"] != "HomeCredit"]
    assert len({r["dataset"] for r in rows}) == 2, "expected the SBA and simulator arms on file"

    # ---------------- Home Credit arm ----------------
    d = np.load(os.path.join(PROC, "hc.npz"), allow_pickle=True)
    X, y, test = d["X"], d["y"].astype(float), d["test"].astype(bool)
    sc = np.load(os.path.join(PROC, "hc_scores.npz"))
    f0 = sc["f0_gbdt"]
    rows += rank_configs(np.nan_to_num(X[~test]), y[~test], np.nan_to_num(X[test]), y[test],
                         f0, "HomeCredit")
    dump("gate_configs.json", rows)

    # ---------------- tau on both arms ----------------
    key = lambda r: (r["objective"], r["num_leaves"])
    sim = {key(r): r["DIR_a07"] for r in rows if r["dataset"] == "simulator"}
    real = {ds: {key(r): r["DIR_a07"] for r in rows if r["dataset"] == ds}
            for ds in ("SBA", "HomeCredit")}
    ks = sorted(sim)
    top3 = sorted(ks, key=lambda k: -sim[k])[:3]
    pairs = [(a, b) for i, a in enumerate(top3) for b in top3[i + 1:]]

    arm = {}
    for ds, rk in real.items():
        tau, p = kendalltau([sim[k] for k in ks], [rk[k] for k in ks])
        agree = sum(1 for a, b in pairs if np.sign(sim[a] - sim[b]) == np.sign(rk[a] - rk[b]))
        arm[ds] = dict(kendall_tau=float(tau), kendall_tau_p=float(p),
                       tau_at_or_above_threshold=bool(tau >= THRESH),
                       top3_sign_agreements=f"{agree} of {len(pairs)}",
                       n_sign_agreements=int(agree), n_sign_pairs=len(pairs),
                       all_signs_agree=bool(agree == len(pairs)),
                       arm_passes=bool(tau >= THRESH and agree == len(pairs)),
                       DIR_by_configuration={f"{o} leaves={l}": rk[(o, l)] for o, l in ks})

    passes = all(a["arm_passes"] for a in arm.values())
    failing = [ds for ds, a in arm.items() if not a["arm_passes"]]
    why = []
    for ds in ("SBA", "HomeCredit"):
        a = arm[ds]
        if not a["tau_at_or_above_threshold"]:
            why.append(f"tau = {a['kendall_tau']:+.2f} on {ds} is below the 0.6 threshold")
        if not a["all_signs_agree"]:
            why.append(f"{a['n_sign_pairs'] - a['n_sign_agreements']} of {a['n_sign_pairs']} "
                       f"pairwise signs among the top three disagree on {ds}")

    out = dict(
        record="AUTHORITATIVE -- supersedes results/gate_status_sba_arm.json, which records "
               "the SBA arm alone",
        source="simulator + SBA National + Home Credit",
        rule=("Section 5.2: the gate passes only if Kendall tau between the simulator's ranking "
              "of the learner configurations and EACH real-data ranking is at least 0.6 and no "
              "pairwise sign among the top three configurations disagrees on either real "
              "dataset."),
        threshold=THRESH,
        n_configurations=len(ks),
        configurations=[f"{o} leaves={l}" for o, l in ks],
        configurations_note="the same six on both arms, as fixed in 10_gate_kendall.py",
        simulator_DIR_by_configuration={f"{o} leaves={l}": sim[(o, l)] for o, l in ks},
        top3_simulator=[f"{o} leaves={l}" for o, l in top3],
        kendall_tau_sba=arm["SBA"]["kendall_tau"],
        kendall_tau_sba_p=arm["SBA"]["kendall_tau_p"],
        kendall_tau_home_credit=arm["HomeCredit"]["kendall_tau"],
        kendall_tau_home_credit_p=arm["HomeCredit"]["kendall_tau_p"],
        top_three_sign_agreement_sba=arm["SBA"]["top3_sign_agreements"],
        top_three_sign_agreement_home_credit=arm["HomeCredit"]["top3_sign_agreements"],
        arms=arm,
        decided=True,
        gate_decision="pass" if passes else "fail",
        failing_arms=failing,
        reasons=why,
        simulator_seven_source_set="admitted" if passes else "withheld",
        consequence=("the simulator's seven-source Saudi attribution is reported"
                     if passes else
                     "no simulator-based ranking is reported and the source-level analysis is "
                     "confined to the real datasets (SBA's three sources and Home Credit's "
                     "four); the simulator's seven-source Saudi attribution is WITHHELD"),
        note=("The gate failing is the pre-specified conservative rule working as designed, not "
              "a failure of the run. Both real arms were computed on the same six configurations "
              "against the same simulator ranking."))
    dump("gate_status.json", out)
    log("GATE DECISION", json.dumps({k: out[k] for k in
                                     ["kendall_tau_sba", "kendall_tau_home_credit",
                                      "top_three_sign_agreement_sba",
                                      "top_three_sign_agreement_home_credit",
                                      "gate_decision", "simulator_seven_source_set"]}, indent=1))


if __name__ == "__main__":
    main()
