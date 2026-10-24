"""Home Credit: Q2, what accepted-loan data identify (Proposition 2).

Mirrors 09_sba_q2.py exactly. application_train.csv records TARGET for every booked
applicant and carries no rejects, so the tuned incumbent's top 70% by score is TREATED as
the observed set A_0 and outcomes outside it are masked for estimation -- the same
construction 09_sba_q2.py applies to SBA. The true mu_out, true M and true DIR are also computed and
reported beside the identified set.

Also extends the Supplementary S5 coverage study to Home Credit's own (n, base rate), as
09_sba_q2.py does for SBA, because the protocol forbids reporting an interval where coverage has not
been established.
"""
from __future__ import annotations

import os
import time

import numpy as np

from common import dump
from src.eval.dir_metric import _approved_index, default_inflow, dir_metric  # noqa: E402

from common import PROC  # the repository root is resolved once, in common.py
A0 = 0.7
R_TARGET = 0.15
T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)


def approved(score, a, seed=0):
    return _approved_index(np.asarray(score, float), a, np.random.default_rng(seed))


def identified(f, f0, y, a, a0=A0, seed=0):
    n = len(y)
    A = set(approved(f, a, seed).tolist())
    A_0 = set(approved(f0, a0, seed).tolist())
    inter = np.array(sorted(A & A_0), int)
    out = np.array(sorted(A - A_0), int)
    s = len(out) / len(A)
    mu_in = float(y[inter].mean()) if len(inter) else float("nan")
    mu_out_true = float(y[out].mean()) if len(out) else float("nan")
    DI0 = default_inflow(f0, y, a, seed=seed)
    W_per_M = s * mu_in / DI0
    Mstar = 1.0 + (1.0 / s) * ((1.0 - R_TARGET) * DI0 / mu_in - 1.0) if s > 0 else float("inf")
    true_M = mu_out_true / mu_in if mu_in > 0 else float("nan")
    DI_true = float(y[np.array(sorted(A), int)].mean())
    return dict(a=a, a0=a0, n=n, swap_share=s, n_swap=len(out), mu_in=mu_in,
                mu_out_true=mu_out_true, DI0=DI0, width_per_unit_M=W_per_M,
                M_star_015=Mstar, true_M=true_M, DI_true=DI_true,
                DIR_true=1.0 - DI_true / DI0,
                set_lo_at_true_M=1.0 - mu_in * (1 + s * (true_M - 1)) / DI0,
                set_hi=1.0 - mu_in / DI0,
                true_DIR_in_set=bool((1.0 - mu_in * (1 + s * (true_M - 1)) / DI0)
                                     <= (1.0 - DI_true / DI0) <= (1.0 - mu_in / DI0) + 1e-12),
                lower_inequality_holds=bool(mu_out_true >= mu_in))


def boot_mstar(f, f0, y, a, n_boot=400, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals, svals, wvals = [], [], []
    for b in range(n_boot):
        ix = rng.integers(0, n, n)
        r = identified(f[ix], f0[ix], y[ix], a, seed=b)
        vals.append(r["M_star_015"]); svals.append(r["swap_share"]); wvals.append(r["width_per_unit_M"])
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return q(vals), q(svals), q(wvals)


def s5_hc_cell(f, f0, y, a=0.6, m=40_000, reps=60, n_boot=60, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    truth = dir_metric(f, f0, y, a, n_tie_draws=1)
    cov = 0
    for r in range(reps):
        ix = rng.choice(n, m, replace=False)
        fs, f0s, ys = f[ix], f0[ix], y[ix]
        bt = np.empty(n_boot)
        for b in range(n_boot):
            jx = rng.integers(0, m, m)
            bt[b] = dir_metric(fs[jx], f0s[jx], ys[jx], a, n_tie_draws=1, seed=b)
        lo, hi = np.percentile(bt, [2.5, 97.5])
        cov += int(lo <= truth <= hi)
    return dict(source="HomeCredit", study="S5_coverage_extension", a=a, population_n=int(n),
                subsample_m=int(m), reps=int(reps), n_boot=int(n_boot),
                base_rate=float(y.mean()), population_DIR=float(truth),
                coverage=cov / reps, nominal=0.95)


def main():
    d = np.load(os.path.join(PROC, "hc_scores.npz"))
    y = d["y_test"].astype(float)
    f = d["f_A+U+I+P"]
    f0 = d["f0_gbdt"]

    cov = s5_hc_cell(f, f0, y)
    dump("HC_S5_coverage.json", cov)
    log("S5 extension on Home Credit", cov["coverage"])

    rows = []
    for a in [0.5, 0.6, 0.7]:
        r = identified(f, f0, y, a)
        r["source"] = "HomeCredit"
        rows.append(r)
        log(f"a={a}", {k: (round(v, 4) if isinstance(v, float) else v)
                       for k, v in r.items() if k in ("swap_share", "mu_in", "DI0", "M_star_015",
                                                      "true_M", "width_per_unit_M")})
    dump("HC_Q2_cells.json", rows)

    r06 = next(r for r in rows if r["a"] == 0.6)
    r07 = next(r for r in rows if r["a"] == 0.7)
    ci_m, ci_s, ci_w = boot_mstar(f, f0, y, 0.6, n_boot=400, seed=0)
    ci_m7, _, _ = boot_mstar(f, f0, y, 0.7, n_boot=400, seed=1)
    log("bootstrap done")
    summ = dict(source="HomeCredit", coverage_established=cov["coverage"] >= 0.90,
                coverage_value=cov["coverage"],
                swap_share_a06=r06["swap_share"], swap_share_a06_ci=ci_s,
                mu_in_a06=r06["mu_in"], DI0_a06=r06["DI0"],
                width_per_unit_M_a06=r06["width_per_unit_M"], width_per_unit_M_a06_ci=ci_w,
                M_star_015_a06=r06["M_star_015"], M_star_015_a06_ci=ci_m,
                swap_share_a07=r07["swap_share"], M_star_015_a07=r07["M_star_015"],
                M_star_015_a07_ci=ci_m7,
                true_M_a06=r06["true_M"], true_M_a07=r07["true_M"],
                true_DIR_a06=r06["DIR_true"], true_DIR_in_set_a06=r06["true_DIR_in_set"],
                lower_inequality_holds_a06=r06["lower_inequality_holds"],
                identified_set_at_M4_a06=[1.0 - r06["mu_in"] * (1 + r06["swap_share"] * 3) / r06["DI0"],
                                          1.0 - r06["mu_in"] / r06["DI0"]])
    dump("HC_Q2_summary.json", summ)
    log("SUMMARY", summ)


if __name__ == "__main__":
    main()
