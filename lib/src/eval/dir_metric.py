"""Default-Inflow-Reduction (DIR) metric and swap-set analysis.

DI(f, a)  = default rate among the top-a fraction of applicants ranked by score f.
DIR(f, a | f0) = 1 - DI(f, a) / DI(f0, a)   -- the "% reduction in new default inflow" KPI.

SCORE CONVENTION (this module is the single source of truth for the repository):
    score = CREDIT QUALITY, higher = better credit = lower risk. The top-a fraction by score is approved.
    Model risk margins (e.g. LightGBM log-odds of default) must be converted with `risk_to_score`
    before entering any function here. `assert_score_convention` fails loudly if the sign is flipped.

Implementation choices:
  * Ties at the threshold are broken RANDOMLY and averaged over `n_tie_draws` draws (was: stable argsort).
  * DIR raises `DegenerateDenominator` when DI(f0, a) is below `min_di_old` instead of clamping to 1e-12;
    `di_difference_pp` (percentage points) is reported alongside DIR.
  * `bootstrap_dir` offers `method="subsample"` (m-out-of-n without replacement) as an alternative to the
    paired i.i.d. bootstrap, because the quantile-thresholded top-k functional is not automatically
    bootstrap-valid; coverage must be checked by simulation (`tests/test_dir_metric.py::test_bootstrap_coverage_smoke`).

All functions operate on held-out data with observed outcomes. Where a function takes `sample_weight`,
the weights can be used to re-weight the held-out sample.
"""
from __future__ import annotations

import numpy as np


class DegenerateDenominator(ValueError):
    pass


def risk_to_score(risk_margin: np.ndarray) -> np.ndarray:
    """Convert a risk margin (higher = more likely to default) to the repo's credit-quality score."""
    return -np.asarray(risk_margin, float)


def assert_score_convention(score: np.ndarray, y: np.ndarray, min_auc: float = 0.5) -> None:
    """Loud failure if `score` is not oriented as credit quality (defaulters should score LOWER).
    Uses a rank-based AUC of (1-y) vs score (pure numpy, Mann-Whitney); raises if AUC < min_auc."""
    y = np.asarray(y, float); score = np.asarray(score, float)
    if y.min() == y.max():
        return
    from scipy.stats import rankdata
    r = rankdata(score)
    n_pos, n_neg = int((y == 0).sum()), int((y == 1).sum())      # "positive" = non-defaulter
    auc = (r[y == 0].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    if auc < min_auc:
        raise ValueError(f"score convention violated: AUC(non-default vs score)={auc:.3f} < {min_auc}. "
                         "Scores must be credit quality (higher = better). Use risk_to_score().")


def _approved_index(score: np.ndarray, approval_rate: float, rng: np.random.Generator) -> np.ndarray:
    n = len(score)
    n_approve = max(1, int(round(approval_rate * n)))
    jitter = rng.random(n)                       # random tie-break (comment 20)
    order = np.lexsort((jitter, -score))
    return order[:n_approve]


def default_inflow(score: np.ndarray, y: np.ndarray, approval_rate: float,
                   sample_weight: np.ndarray | None = None, n_tie_draws: int = 5, seed: int = 0) -> float:
    """Default rate among approved loans when approving the top `approval_rate` by credit-quality score.
    Averaged over `n_tie_draws` random tie-break draws."""
    score = np.asarray(score, float)
    y = np.asarray(y, float)
    w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_tie_draws):
        idx = _approved_index(score, approval_rate, rng)
        vals.append(np.sum(w[idx] * y[idx]) / np.sum(w[idx]))
    return float(np.mean(vals))


def dir_metric(score_new: np.ndarray, score_old: np.ndarray, y: np.ndarray, approval_rate: float,
               sample_weight: np.ndarray | None = None, min_di_old: float = 1e-3, **kw) -> float:
    """DIR = 1 - DI_new / DI_old. Raises DegenerateDenominator if DI_old < min_di_old (comment 21)."""
    di_new = default_inflow(score_new, y, approval_rate, sample_weight, **kw)
    di_old = default_inflow(score_old, y, approval_rate, sample_weight, **kw)
    if di_old < min_di_old:
        raise DegenerateDenominator(f"DI(f0, a={approval_rate})={di_old:.5f} < {min_di_old}; DIR undefined. "
                                    "Report di_difference_pp instead.")
    return 1.0 - di_new / di_old


def di_difference_pp(score_new, score_old, y, approval_rate, sample_weight=None, **kw) -> float:
    """DI_old - DI_new in percentage points (stable where DIR is not)."""
    return 100.0 * (default_inflow(score_old, y, approval_rate, sample_weight, **kw)
                    - default_inflow(score_new, y, approval_rate, sample_weight, **kw))


def swap_set(score_new: np.ndarray, score_old: np.ndarray, y: np.ndarray, approval_rate: float,
             seed: int = 0) -> dict:
    """Swap-set decomposition at a fixed approval rate (one random tie-break draw).
    Returns default rates in the four cells: approved by both, new-only, old-only, plus swap size."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    k = max(1, int(round(approval_rate * len(y))))
    new_set = set(_approved_index(np.asarray(score_new, float), approval_rate, rng))
    old_set = set(_approved_index(np.asarray(score_old, float), approval_rate, rng))
    both = np.array(sorted(new_set & old_set), int)
    new_only = np.array(sorted(new_set - old_set), int)
    old_only = np.array(sorted(old_set - new_set), int)

    def rate(ix):
        return float(y[ix].mean()) if len(ix) else float("nan")

    return {"approval_rate": approval_rate, "n_swap": int(len(new_only)), "swap_share": len(new_only) / k,
            "dr_both": rate(both), "dr_new_only": rate(new_only), "dr_old_only": rate(old_only),
            "di_new": rate(np.array(sorted(new_set), int)), "di_old": rate(np.array(sorted(old_set), int))}


def bootstrap_dir(score_new, score_old, y, approval_rate, n_boot=1000, seed=0, sample_weight=None,
                  method: str = "paired", subsample_frac: float = 0.5) -> tuple[float, float, float]:
    """Point estimate and percentile 95% CI of DIR.
    method='paired'   : paired i.i.d. bootstrap over applicants (with replacement; duplicates enter top-k).
    method='subsample': m-out-of-n subsampling without replacement, m = subsample_frac * n, rescaled.
    Validity for the top-k functional is NOT automatic; see tests for the coverage smoke check."""
    rng = np.random.default_rng(seed)
    score_new, score_old, y = map(lambda v: np.asarray(v, float), (score_new, score_old, y))
    n = len(y)
    est = dir_metric(score_new, score_old, y, approval_rate, sample_weight)
    boots = np.empty(n_boot)
    m = int(subsample_frac * n)
    for b in range(n_boot):
        ix = rng.integers(0, n, n) if method == "paired" else rng.choice(n, m, replace=False)
        sw = None if sample_weight is None else np.asarray(sample_weight)[ix]
        boots[b] = dir_metric(score_new[ix], score_old[ix], y[ix], approval_rate, sw, seed=b)
    if method == "subsample":   # rescale deviations by sqrt(m/n)
        boots = est + (boots - est) * np.sqrt(m / n)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return est, float(lo), float(hi)


def dir_curve(score_new, score_old, y, approval_rates=(0.5, 0.6, 0.7, 0.8, 0.9), **kw):
    out = {}
    for a in approval_rates:
        try:
            out[a] = bootstrap_dir(score_new, score_old, y, a, **kw)
        except DegenerateDenominator:
            out[a] = (float("nan"), float("nan"), float("nan"))
    return out


if __name__ == "__main__":
    # Demonstration (NOT a result): AUC -> DIR non-monotonicity on synthetic scores
    # (a known property, shown as motivation only).
    rng = np.random.default_rng(1)
    n = 20000
    y = (rng.random(n) < 0.08).astype(float)
    latent = -2.0 * y + rng.normal(size=n)
    f_old = latent + rng.normal(scale=1.5, size=n)
    f_A = latent + rng.normal(scale=1.0, size=n)
    f_B = np.where(latent > np.quantile(latent, 0.4), latent + rng.normal(scale=0.3, size=n), rng.normal(scale=3.0, size=n))
    for name, f in [("old", f_old), ("A", f_A), ("B", f_B)]:
        assert_score_convention(f, y)
        print(f"{name}:", {a: round(dir_metric(f, f_old, y, a), 3) for a in (0.5, 0.6, 0.7, 0.8, 0.9)})
    print(swap_set(f_A, f_old, y, 0.7))
