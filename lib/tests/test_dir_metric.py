"""Known-answer, tie, degenerate-a and bootstrap tests for eval/dir_metric.py."""
import numpy as np
import pytest

from src.eval.dir_metric import (DegenerateDenominator, assert_score_convention, bootstrap_dir, default_inflow,
                                 di_difference_pp, dir_metric, risk_to_score, swap_set)


def test_known_answer_perfect_ranker():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], float)
    score = np.array([8, 7, 6, 5, 4, 3, 2, 1], float)   # perfect: non-defaulters highest
    assert default_inflow(score, y, 0.5) == 0.0
    assert default_inflow(score, y, 1.0) == 0.5


def test_known_answer_dir():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], float)
    f_new = np.array([8, 7, 6, 5, 4, 3, 2, 1], float)
    f_old = np.array([8, 1, 6, 2, 7, 3, 5, 4], float)    # approves two defaulters at a=0.5
    assert default_inflow(f_old, y, 0.5) == pytest.approx(0.5)
    assert dir_metric(f_new, f_old, y, 0.5) == pytest.approx(1.0)
    assert di_difference_pp(f_new, f_old, y, 0.5) == pytest.approx(50.0)


def test_ties_are_random_and_averaged():
    y = np.array([1, 0, 1, 0], float)
    score = np.zeros(4)                                   # all tied
    vals = [default_inflow(score, y, 0.5, n_tie_draws=1, seed=s) for s in range(20)]
    assert min(vals) < max(vals)                          # random tie-break moves DI
    avg = default_inflow(score, y, 0.5, n_tie_draws=200, seed=0)
    assert abs(avg - 0.5) < 0.1                           # averages toward the base rate


def test_degenerate_denominator_raises():
    y = np.array([0, 0, 0, 0, 1, 1], float)
    f_old = np.array([6, 5, 4, 3, 2, 1], float)           # DI_old = 0 at a=0.5
    with pytest.raises(DegenerateDenominator):
        dir_metric(f_old, f_old, y, 0.5)


def test_tiny_approval_rate_approves_one():
    y = np.array([1, 0, 0], float)
    assert default_inflow(np.array([1, 2, 3.]), y, 0.01) == 0.0


def test_swap_set_empty_when_identical():
    y = np.array([0, 1, 0, 1], float)
    s = np.array([4, 3, 2, 1.])
    out = swap_set(s, s, y, 0.5)
    assert out["n_swap"] == 0 and np.isnan(out["dr_new_only"])


def test_score_convention_assertion():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.1).astype(float)
    risk = 2 * y + rng.normal(size=2000)
    with pytest.raises(ValueError):
        assert_score_convention(risk, y)                  # risk passed as score -> loud failure
    assert_score_convention(risk_to_score(risk), y)       # converted -> passes


def test_bootstrap_methods_run():
    rng = np.random.default_rng(1)
    y = (rng.random(3000) < 0.1).astype(float)
    f_old = -y + rng.normal(size=3000)
    f_new = -1.5 * y + rng.normal(size=3000)
    for m in ("paired", "subsample"):
        est, lo, hi = bootstrap_dir(f_new, f_old, y, 0.7, n_boot=50, method=m)
        assert lo <= est <= hi


@pytest.mark.slow
def test_bootstrap_coverage_smoke():
    """Coverage of the paired bootstrap for the top-k functional (comment 18). Reports, does not assert
    tightly: the point is to make the check exist and be run before any CI is reported."""
    rng = np.random.default_rng(2)
    n, reps, cover = 2000, 60, 0
    # population DIR by a large sample
    yb = (rng.random(200000) < 0.1).astype(float)
    fo = -yb + rng.normal(size=200000); fn = -1.5 * yb + rng.normal(size=200000)
    truth = dir_metric(fn, fo, yb, 0.7)
    for r in range(reps):
        y = (rng.random(n) < 0.1).astype(float)
        f_old = -y + rng.normal(size=n); f_new = -1.5 * y + rng.normal(size=n)
        _, lo, hi = bootstrap_dir(f_new, f_old, y, 0.7, n_boot=100, seed=r)
        cover += lo <= truth <= hi
    assert cover / reps > 0.8, f"coverage {cover/reps:.2f} far below nominal 0.95"
