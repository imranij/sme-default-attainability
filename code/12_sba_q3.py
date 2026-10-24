"""SBA National: Q3, source-level value of information (Section 4.3).

Sources F / G / B were fixed in 07_sba_prepare.py BEFORE any fit. Both characteristic
functions of Section 4.3 are computed at a = 0.7 against the tuned GBDT incumbent:

  v_ach(C)  = DIR(f_C, a | f0)          achieved by one learner
  v*(C)     = DIR*(C, a | f0)           the ceiling of Section 3.2
  v(empty)  = 1 - pi / DI(f0, a)        explicit empty coalition (random ranking)

Bootstrap: the TEST window is resampled with the fitted scores and the cross-fitted
eta_hat held fixed, which propagates sampling noise in the KPI but not fit-to-fit
variance. Pairwise Shapley-Taylor interaction indices and the leave-one-source-out
ranking are reported beside the Shapley ranking, as Section 4.3 requires.
"""
from __future__ import annotations

import itertools
import math
import os

import numpy as np

from common import A_OP, dump
from src.eval.dir_metric import _approved_index, default_inflow  # noqa: E402

from common import PROC  # the repository root is resolved once, in common.py
SRC = ["F", "G", "B"]
NAME = {"F": "firmographics", "G": "guarantee", "B": "bank"}
K = 3


def tag_of(C):
    return "+".join(s for s in SRC if s in C) if C else "EMPTY"


def tail_mean(eta, a, ix=None):
    e = eta if ix is None else eta[ix]
    k = max(1, int(round(a * len(e))))
    return float(np.sort(e)[:k].mean())


def shapley(v):
    """Exact Shapley value for k=3 from a dict {frozenset: value}."""
    phi = {}
    others = lambda j: [s for s in SRC if s != j]
    for j in SRC:
        tot = 0.0
        for r in range(K):
            for C in itertools.combinations(others(j), r):
                w = math.factorial(len(C)) * math.factorial(K - len(C) - 1) / math.factorial(K)
                tot += w * (v[frozenset(C) | {j}] - v[frozenset(C)])
        phi[j] = tot
    return phi


def shapley_taylor_pairs(v):
    """Second-order Shapley-Taylor interaction index for k=3 (Sundararajan et al. 2020)."""
    out = {}
    for i, j in itertools.combinations(SRC, 2):
        rest = [s for s in SRC if s not in (i, j)]
        tot = 0.0
        for r in range(len(rest) + 1):
            for C in itertools.combinations(rest, r):
                Cs = frozenset(C)
                delta = (v[Cs | {i, j}] - v[Cs | {i}] - v[Cs | {j}] + v[Cs])
                w = (K - 1) / (K - 1) * (math.factorial(len(C)) * math.factorial(K - 2 - len(C))
                                         / math.factorial(K - 1))
                tot += w * delta
        out[f"{i}x{j}"] = tot
    return out


def main():
    sc = np.load(os.path.join(PROC, "sba_scores.npz"))
    y = sc["y_test"].astype(float)
    f0 = sc["f0_gbdt"]
    n = len(y)
    subsets = [frozenset(c) for r in range(K + 1) for c in itertools.combinations(SRC, r)]

    def values(ix=None):
        yy = y if ix is None else y[ix]
        ff0 = f0 if ix is None else f0[ix]
        DI0 = default_inflow(ff0, yy, A_OP, n_tie_draws=1)
        pi = float(yy.mean())
        v_ach, v_star = {}, {}
        for C in subsets:
            t = tag_of(C)
            if not C:
                v_ach[C] = 1.0 - pi / DI0
                v_star[C] = 1.0 - pi / DI0
                continue
            f = sc[f"f_{t}"] if ix is None else sc[f"f_{t}"][ix]
            v_ach[C] = 1.0 - default_inflow(f, yy, A_OP, n_tie_draws=1) / DI0
            eta = sc[f"eta_{t}"] if ix is None else sc[f"eta_{t}"][ix]
            v_star[C] = 1.0 - tail_mean(eta, A_OP) / DI0
        return v_ach, v_star, DI0, pi

    v_ach, v_star, DI0, pi = values()
    phi_ach, phi_star = shapley(v_ach), shapley(v_star)
    inter_star = shapley_taylor_pairs(v_star)
    inter_ach = shapley_taylor_pairs(v_ach)
    full = frozenset(SRC)
    loso_star = {j: v_star[full] - v_star[full - {j}] for j in SRC}
    loso_ach = {j: v_ach[full] - v_ach[full - {j}] for j in SRC}

    rank = lambda d: [k for k, _ in sorted(d.items(), key=lambda kv: -kv[1])]
    # ---- bootstrap over the test window ------------------------------------
    B = 300
    rng = np.random.default_rng(0)
    rank_counts = {j: np.zeros(K, int) for j in SRC}
    rank_counts_ach = {j: np.zeros(K, int) for j in SRC}
    top_star, top_ach, phis = [], [], {j: [] for j in SRC}
    for b in range(B):
        ix = rng.integers(0, n, n)
        va, vs, _, _ = values(ix)
        ps, pa = shapley(vs), shapley(va)
        rs, ra = rank(ps), rank(pa)
        for r, j in enumerate(rs):
            rank_counts[j][r] += 1
        for r, j in enumerate(ra):
            rank_counts_ach[j][r] += 1
        top_star.append(rs[0]); top_ach.append(ra[0])
        for j in SRC:
            phis[j].append(ps[j])
    top1 = rank(phi_star)[0]
    prob_top = float(np.mean([t == top1 for t in top_star]))
    top1a = rank(phi_ach)[0]
    prob_top_ach = float(np.mean([t == top1a for t in top_ach]))
    # ranks that hold across draws: a rank position whose modal source holds in >=90% of draws
    holds, reshuffles = [], []
    for r in range(K):
        counts = {j: int(rank_counts[j][r]) for j in SRC}
        best = max(counts, key=counts.get)
        (holds if counts[best] / B >= 0.90 else reshuffles).append(
            dict(rank=r + 1, modal_source=NAME[best], probability=counts[best] / B))

    out = dict(
        source="SBA", a=A_OP, DI0_tuned=DI0, base_rate_test=pi, n_test=int(n),
        sources={k: NAME[k] for k in SRC},
        v_attainable={tag_of(C): v_star[C] for C in subsets},
        v_achieved={tag_of(C): v_ach[C] for C in subsets},
        shapley_attainable={NAME[k]: v for k, v in phi_star.items()},
        shapley_achieved={NAME[k]: v for k, v in phi_ach.items()},
        shapley_attainable_ci={NAME[j]: [float(np.percentile(phis[j], 2.5)),
                                         float(np.percentile(phis[j], 97.5))] for j in SRC},
        ranking_attainable=[NAME[j] for j in rank(phi_star)],
        ranking_achieved=[NAME[j] for j in rank(phi_ach)],
        top_source_attainable=NAME[top1], top_source_value_attainable=phi_star[top1],
        top_rank_probability_attainable=prob_top,
        top_source_achieved=NAME[top1a], top_rank_probability_achieved=prob_top_ach,
        rank_probabilities={NAME[j]: (rank_counts[j] / B).tolist() for j in SRC},
        ranks_that_hold=holds, ranks_that_reshuffle=reshuffles,
        negative_sources_achieved=[NAME[j] for j in SRC if phi_ach[j] < 0],
        negative_sources_attainable=[NAME[j] for j in SRC if phi_star[j] < 0],
        distance_attainable_minus_achieved={NAME[j]: phi_star[j] - phi_ach[j] for j in SRC},
        distance_total=float(sum(phi_star.values()) - sum(phi_ach.values())),
        interactions_attainable={k.replace("F", "firmographics").replace("G", "guarantee")
                                 .replace("B", "bank"): v for k, v in inter_star.items()},
        interactions_achieved={k.replace("F", "firmographics").replace("G", "guarantee")
                               .replace("B", "bank"): v for k, v in inter_ach.items()},
        loso_attainable={NAME[j]: v for j, v in loso_star.items()},
        loso_achieved={NAME[j]: v for j, v in loso_ach.items()},
        loso_ranking_attainable=[NAME[j] for j in rank(loso_star)],
        sources_whose_rank_differs=[NAME[j] for j in SRC
                                    if rank(loso_star).index(j) != rank(phi_star).index(j)],
        n_bootstrap=B,
        simulator_seven_source="withheld (the sim-to-real gate is decided in 19_gate_hc.py)")
    lp = max(inter_star.items(), key=lambda kv: abs(kv[1]))
    out["largest_interacting_pair"] = lp[0]
    out["largest_interaction"] = lp[1]
    dump("SBA_Q3_shapley.json", out)
    print({k: out[k] for k in ["ranking_attainable", "top_source_attainable",
                               "top_source_value_attainable", "top_rank_probability_attainable",
                               "ranking_achieved", "negative_sources_achieved",
                               "largest_interacting_pair", "largest_interaction",
                               "sources_whose_rank_differs"]})


if __name__ == "__main__":
    main()
