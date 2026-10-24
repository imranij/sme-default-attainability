"""Gradient tests for models/dial_objective.py.

Two separate facts are tested:
  1. The analytic gradient equals the finite-difference gradient of the loss WITH TAU HELD FIXED (assertion).
  2. The tau-constant approximation omits a term whose size is quantified relative to the gradient
     (reported; a loose bound is asserted so a regression is noticed).
Plus: non-defaulters receive non-zero gradient (degeneracy fix), and the torch and LightGBM paths
optimise the same functional.
"""
import numpy as np
import pytest

from src.models.dial_objective import (_sigmoid, dial_loss_np, lightgbm_dial_objective, soft_quantile_np,
                                       surrogate_terms, tau_gradient_correction)


class _TD:
    def __init__(self, y):
        self.y = y

    def get_label(self):
        return self.y


def _setup(seed=0, n=400, a=0.7, kappa=0.5, lam=0.5, beta=1.0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.1).astype(float)
    f = rng.normal(size=n)
    return y, f, a, kappa, lam, beta


def test_gradient_matches_finite_difference_with_tau_fixed():
    y, f, a, kappa, lam, beta = _setup()
    grad, _ = lightgbm_dial_objective(a, kappa, lam, beta)(f, _TD(y))
    tau = soft_quantile_np(f, 1 - a, kappa)
    n = len(f)

    def loss_tau_fixed(fv):
        s = _sigmoid((tau - fv) / kappa)
        p = _sigmoid(fv)
        bce = -(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
        return np.mean((y - beta * (1 - y)) * s) + lam * np.mean(bce)

    eps = 1e-5
    for i in [int(np.argmax(y)), int(np.argmin(y))]:          # one defaulter, one non-defaulter
        fp, fm = f.copy(), f.copy(); fp[i] += eps; fm[i] -= eps
        fd = (loss_tau_fixed(fp) - loss_tau_fixed(fm)) / (2 * eps)
        assert grad[i] == pytest.approx(fd, rel=1e-4, abs=1e-8)


def test_tau_bias_is_quantified():
    y, f, a, kappa, lam, beta = _setup()
    grad, _ = lightgbm_dial_objective(a, kappa, lam, beta)(f, _TD(y))
    corr = tau_gradient_correction(f, y, a, kappa, beta)
    ratio = np.abs(corr).sum() / np.abs(grad).sum()
    # the full-loss finite difference should equal grad + corr
    eps = 1e-4; i = int(np.argmax(y))
    fp, fm = f.copy(), f.copy(); fp[i] += eps; fm[i] -= eps
    fd_full = (dial_loss_np(fp, y, a, kappa, lam, beta, iters=80) - dial_loss_np(fm, y, a, kappa, lam, beta, iters=80)) / (2 * eps)
    assert fd_full == pytest.approx(grad[i] + corr[i], rel=2e-2, abs=1e-6)
    assert abs(fd_full - grad[i]) > abs(fd_full - (grad[i] + corr[i]))   # the correction moves toward truth
    print(f"tau-correction / gradient L1 ratio = {ratio:.4f}")
    assert ratio < 0.5, "tau-constant approximation omits more than half the gradient mass; revisit"


def test_non_defaulters_get_gradient():
    y, f, a, kappa, lam, beta = _setup()
    _, _, _, g_di, _ = surrogate_terms(f, y, a, kappa, beta)
    assert np.abs(g_di[y == 0]).max() > 0                     # degeneracy fix (comment 30)
    _, _, _, g_di0, _ = surrogate_terms(f, y, a, kappa, 0.0)  # beta=0 reproduces the original degenerate term
    assert np.abs(g_di0[y == 0]).max() == 0


def test_hessian_positive():
    y, f, a, kappa, lam, beta = _setup()
    for h in ("gn", "abs"):
        _, hess = lightgbm_dial_objective(a, kappa, lam, beta, hessian=h)(f, _TD(y))
        assert (hess > 0).all()


def test_torch_and_lightgbm_agree():
    torch = pytest.importorskip("torch")
    from src.models.dial_objective import dial_loss_torch
    y, f, a, kappa, lam, beta = _setup()
    l_np = dial_loss_np(f, y, a, kappa, lam, beta)
    l_t = dial_loss_torch(torch.tensor(f, dtype=torch.float64), torch.tensor(y, dtype=torch.float64), a, kappa, lam, beta)
    assert float(l_t) == pytest.approx(l_np, rel=1e-6)
