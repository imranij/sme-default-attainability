"""Berka sequence demonstration, the top-a bootstrap coverage study
(Supplementary S5), the minimum detectable difference (Supplementary S14), and a
placeholder for the sim-to-real gate (Section 5.2).

Berka carries no KPI claim (see the dataset table of the paper); it is used only to show that the
sequence pipeline runs on real monthly transaction data.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold

from common import A_OP, DATA, dump, known_eta_dgp
from src.eval.dir_metric import (bootstrap_dir, default_inflow, dir_metric,  # noqa: E402
                                 risk_to_score, swap_set)

BERKA = os.path.join(DATA, "berka")


# --------------------------------------------------------------------------- Berka
def berka():
    loan = pd.read_csv(os.path.join(BERKA, "loan.asc"), sep=";")
    trans = pd.read_csv(os.path.join(BERKA, "trans.asc"), sep=";", low_memory=False)
    acct = pd.read_csv(os.path.join(BERKA, "account.asc"), sep=";")
    # Label: status B (finished, not repaid) and D (running, in debt) are the default classes.
    loan["y"] = loan.status.isin(["B", "D"]).astype(float)

    # monthly cash-flow sequence per account, restricted to transactions BEFORE the loan date
    trans = trans.merge(loan[["account_id", "date"]].rename(columns={"date": "loan_date"}),
                        on="account_id", how="inner")
    pre = trans[trans.date < trans.loan_date].copy()
    pre["ym"] = pre.date // 100
    pre["signed"] = np.where(pre.type.astype(str).str.startswith("V"), -pre.amount, pre.amount)
    g = pre.groupby(["account_id", "ym"]).agg(inflow=("amount", "sum"), n_txn=("amount", "size"),
                                              bal=("balance", "last"), net=("signed", "sum"))
    g = g.reset_index()
    feat = g.groupby("account_id").agg(
        seq_months=("ym", "nunique"), inflow_mean=("inflow", "mean"), inflow_std=("inflow", "std"),
        net_mean=("net", "mean"), net_std=("net", "std"), bal_min=("bal", "min"),
        bal_mean=("bal", "mean"), bal_last=("bal", "last"), ntxn_mean=("n_txn", "mean")).reset_index()
    df = loan.merge(feat, on="account_id", how="left").merge(
        acct[["account_id", "frequency"]], on="account_id", how="left")
    df = pd.get_dummies(df, columns=["frequency"], dtype=float)
    Xcols = [c for c in df.columns if c not in ("loan_id", "account_id", "status", "y", "date")]
    X = df[Xcols].astype(float).fillna(0.0).to_numpy()
    y = df.y.to_numpy()

    # out-of-fold scores: repeated stratified CV (no usable time axis, per the protocol)
    oof = np.zeros(len(y))
    for seed in range(5):
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            m = LGBMClassifier(n_estimators=120, learning_rate=0.06, num_leaves=7,
                               min_child_samples=20, verbose=-1, n_jobs=1).fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
        oof += p / 5
    f_new = risk_to_score(oof)
    # incumbent: L2-logistic on the static loan fields only (amount, duration, payments)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    Xs = df[["amount", "duration", "payments"]].astype(float).to_numpy()
    p0 = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xs, y):
        m0 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(Xs[tr], y[tr])
        p0[te] = m0.predict_proba(Xs[te])[:, 1]
    f_old = risk_to_score(p0)

    out = dict(source="berka", dataset="Berka PKDD'99", n_loans=int(len(df)),
               n_default=int(y.sum()), base_rate=float(y.mean()),
               n_trans_rows=int(len(trans)), median_pre_months=float(df.seq_months.median()),
               sequence_pipeline_ran=True)
    for a in [0.5, 0.6, 0.7, 0.8, 0.9]:
        est, lo, hi = bootstrap_dir(f_new, f_old, y, a, n_boot=300, seed=0)
        out[f"DI_new_a{a}"] = default_inflow(f_new, y, a)
        out[f"DI_old_a{a}"] = default_inflow(f_old, y, a)
        out[f"DIR_a{a}"] = est
        out[f"DIR_ci_a{a}"] = [lo, hi]
    out["swap_a0.6"] = swap_set(f_new, f_old, y, 0.6)
    # minimum detectable difference (S6): sd of DIR under the null, bootstrapped
    rng = np.random.default_rng(0)
    null = [dir_metric(f_old[rng.integers(0, len(y), len(y))], f_old, y, A_OP) for _ in range(200)]
    out["mdd_DIR_a0.7"] = float(2.8 * np.std(null))
    return out


# ------------------------------------------------------- top-a bootstrap coverage (S5)
def coverage_study():
    rows = []
    for n in [600, 5_000]:
        for br in [0.02, 0.08]:
            X, y, eta = known_eta_dgp(n, 10, br, 1.0, seed=99)
            if y.sum() < 10:
                continue
            # two fixed scores of differing quality; the TRUE DIR is computed on eta
            rng = np.random.default_rng(7)
            f_old = -(eta + rng.normal(0, 0.06, n))
            f_new = -(eta + rng.normal(0, 0.02, n))
            k = max(1, int(round(A_OP * n)))
            true_dir = 1 - eta[np.argsort(-f_new)[:k]].mean() / eta[np.argsort(-f_old)[:k]].mean()
            for method in ["paired", "subsample"]:
                hit = 0
                R = 80
                for r in range(R):
                    ix = np.random.default_rng(1000 + r).integers(0, n, n)
                    try:
                        _, lo, hi = bootstrap_dir(f_new[ix], f_old[ix], y[ix], A_OP,
                                                  n_boot=60, seed=r, method=method)
                    except Exception:
                        continue
                    hit += int(lo <= true_dir <= hi)
                rows.append(dict(source="simulator", study="S5_bootstrap_coverage", n=n,
                                 base_rate=br, method=method, true_DIR=float(true_dir),
                                 coverage=hit / R, nominal=0.95, reps=R))
                print(rows[-1], flush=True)
    by = {}
    for r in rows:
        by.setdefault(r["method"], []).append(abs(r["coverage"] - 0.95))
    best = min(by, key=lambda m: float(np.mean(by[m])))
    summary = dict(source="simulator", chosen_method=best,
                   coverage_of_chosen=float(np.mean([r["coverage"] for r in rows
                                                     if r["method"] == best])),
                   cells_below_nominal=[f"n={r['n']}, base rate {r['base_rate']:.0%}"
                                        for r in rows if r["method"] == best and r["coverage"] < 0.90],
                   per_cell=rows)
    return rows, summary


def main():
    t0 = time.time()
    b = berka()
    dump("berka_sequence_demo.json", b)
    print("berka done", time.time() - t0, flush=True)
    rows, summ = coverage_study()
    dump("S5_bootstrap_coverage.json", rows)
    dump("S5_bootstrap_summary.json", summ)
    gate = dict(
        source="none", decided=False,
        reason=("The gate of Section 5.2 compares the simulator's ranking of the learner "
                "configurations with their rankings on SBA National and Home Credit. This "
                "step runs before either real dataset is ranked, so the gate is recorded as "
                "undecided here; 10_gate_kendall.py and 19_gate_hc.py decide it."),
        kendall_tau_sba=None, kendall_tau_home_credit=None,
        top_three_sign_agreement=None,
        simulator_seven_source_set="withheld")
    dump("gate_status.json", gate)
    print("done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
