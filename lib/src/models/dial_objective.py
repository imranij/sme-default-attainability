"""DIAL: rate-constrained default-inflow objective (a differentiable surrogate of precision-at-k learning).

Smooth surrogate of the default inflow among the approved top-a fraction:

    L(f) = (1/n) sum_i [ y_i * s_i  -  beta * (1 - y_i) * s_i ]  +  lambda * BCE(f)
    s_i  = sigmoid((tau_a - f_i) / kappa)          soft "approved" indicator (approved iff risk f_i <= tau_a)

Implementation notes:
  * SIGN CONVENTION: inside this module f is a RISK margin (higher = more likely to default), which is what
    LightGBM's binary objective produces. Approval = lowest-a fraction of f. Convert to the repo's
    credit-quality score with `eval.dir_metric.risk_to_score(f) = -f` before any DIR computation.
  * DEGENERACY FIX: the original objective sum_i y_i * s_i has zero gradient for every
    non-defaulter, so it only pushed defaulters out of the approved set and did nothing to pull good
    applicants in. The symmetric term  -beta*(1-y_i)*s_i  rewards approving non-defaulters (beta = 1 by
    default). With the approved fraction pinned at a by tau_a, the objective is the soft version of
    "minimise defaults among approved at fixed rate", i.e. precision@k learning (Narasimhan & Agarwal;
    Kar et al.; Boyd et al.; see the related-work section of the paper).
  * tau_a IS TREATED AS CONSTANT in the gradient in BOTH implementations (LightGBM and torch). The
    implicit-function correction d tau/d f_i = s_i(1-s_i) / sum_j s_j(1-s_j) is provided by
    `tau_gradient_correction` and its magnitude is quantified in tests (not silently ignored).
  * The Hessian option `hessian="gn"` uses the positive Gauss-Newton-like curvature s(1-s)/kappa^2;
    `hessian="abs"` is the original |h| heuristic, retained for comparison and documented as a heuristic.
  * The torch loss optimises the SAME functional as the LightGBM path (mean of the surrogate, not a
    ratio); `tests/test_dial_objective.py::test_torch_and_lightgbm_agree` checks this.
"""
from __future__ import annotations

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def soft_quantile_np(f: np.ndarray, q: float, kappa: float, iters: int = 30) -> float:
    """Sigmoid-relaxed quantile: solve mean(sigmoid((f - t)/kappa)) = 1 - q for t by bisection.
    NOTE (comment 31): 30 bisection passes over the full vector per call; called once per boosting round."""
    lo, hi = float(f.min()) - 5 * kappa, float(f.max()) + 5 * kappa
    for _ in range(iters):
        t = 0.5 * (lo + hi)
        frac_above = np.mean(_sigmoid((f - t) / kappa))
        if frac_above > 1 - q:
            lo = t
        else:
            hi = t
    return 0.5 * (lo + hi)


def surrogate_terms(f: np.ndarray, y: np.ndarray, approval_rate: float, kappa: float, beta: float, iters: int = 30):
    """Return (tau, s, per-record surrogate value, d/df, d2/df2) with tau treated as constant."""
    tau = soft_quantile_np(f, 1 - approval_rate, kappa, iters)
    s = _sigmoid((tau - f) / kappa)                     # approved iff f <= tau
    coef = y - beta * (1 - y)                            # +1 for defaulters, -beta for non-defaulters
    val = coef * s
    ds = -s * (1 - s) / kappa                            # d s / d f
    d2s = s * (1 - s) * (2 * s - 1) / kappa**2           # d2 s / d f2
    return tau, s, val, coef * ds, coef * d2s


def tau_gradient_correction(f: np.ndarray, y: np.ndarray, approval_rate: float, kappa: float, beta: float) -> np.ndarray:
    """Implicit-function term omitted from the gradient: dL/dtau * dtau/df_i.
    With mean(sigmoid((f-tau)/kappa)) fixed, dtau/df_i = w_i / sum_j w_j with w = s(1-s).
    dL/dtau = mean(coef * s(1-s)/kappa). Returned so its size can be REPORTED (comment 28)."""
    tau = soft_quantile_np(f, 1 - approval_rate, kappa)
    s = _sigmoid((tau - f) / kappa)
    coef = y - beta * (1 - y)
    w = s * (1 - s)
    dL_dtau = np.mean(coef * w / kappa)
    return dL_dtau * w / max(w.sum(), 1e-12)


def lightgbm_dial_objective(approval_rate: float, kappa: float, lam: float, beta: float = 1.0,
                            hessian: str = "gn"):
    """LightGBM custom objective closure fobj(preds, train_data). preds are RISK margins."""

    def fobj(preds, train_data):
        y = train_data.get_label().astype(float)
        f = preds.astype(float)
        n = len(f)
        _, s, _, g_di, h_di = surrogate_terms(f, y, approval_rate, kappa, beta)
        g_di, h_di = g_di / n, h_di / n                  # mean, to match the torch path
        p = _sigmoid(f)
        g_bce, h_bce = (p - y) / n, p * (1 - p) / n
        grad = g_di + lam * g_bce
        if hessian == "gn":
            h_pos = np.abs(y - beta * (1 - y)) * s * (1 - s) / kappa**2 / n
        elif hessian == "abs":
            h_pos = np.abs(h_di)
        else:
            raise ValueError(hessian)
        hess = h_pos + lam * h_bce + 1e-6
        return grad, hess

    return fobj


def dial_loss_np(f, y, approval_rate, kappa, lam, beta=1.0, iters: int = 30) -> float:
    """Scalar loss (tau solved at f) matching the LightGBM path; used by tests."""
    _, _, val, _, _ = surrogate_terms(f, y, approval_rate, kappa, beta, iters)
    p = _sigmoid(f)
    bce = -(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
    return float(np.mean(val) + lam * np.mean(bce))


def anneal(step: int, total: int, start: float, end: float) -> float:
    """Linear annealing schedule for kappa and lambda. Schedules are hyperparameters and are swept
    (comment 34); nothing here is tuned."""
    t = min(1.0, step / max(1, total))
    return start + t * (end - start)


try:  # torch is optional at import time
    import torch

    def soft_quantile_torch(f: "torch.Tensor", q: float, kappa: float, iters: int = 30):
        """Bisection for tau; returns a DETACHED tensor: tau is constant w.r.t. f (documented; comment 32).
        For mini-batch training use `EMAQuantile` so tau estimates the population quantile (comment 31)."""
        fd = f.detach()
        lo, hi = fd.min() - 5 * kappa, fd.max() + 5 * kappa
        for _ in range(iters):
            t = 0.5 * (lo + hi)
            frac_above = torch.sigmoid((fd - t) / kappa).mean()
            lo, hi = torch.where(frac_above > 1 - q, t, lo), torch.where(frac_above > 1 - q, hi, t)
        return 0.5 * (lo + hi)

    class EMAQuantile:
        """Running estimate of tau across mini-batches (exponential moving average)."""
        def __init__(self, momentum: float = 0.9):
            self.m, self.value = momentum, None

        def update(self, f, q, kappa):
            t = soft_quantile_torch(f, q, kappa)
            self.value = t if self.value is None else self.m * self.value + (1 - self.m) * t
            return self.value

    def dial_loss_torch(f: "torch.Tensor", y: "torch.Tensor", approval_rate: float, kappa: float, lam: float,
                        beta: float = 1.0, tau: "torch.Tensor | None" = None) -> "torch.Tensor":
        """Same functional as the LightGBM path: mean[(y - beta(1-y)) * s] + lam * BCE, tau constant."""
        if tau is None:
            tau = soft_quantile_torch(f, 1 - approval_rate, kappa)
        s = torch.sigmoid((tau - f) / kappa)
        coef = y - beta * (1 - y)
        di_term = (coef * s).mean()
        bce = torch.nn.functional.binary_cross_entropy_with_logits(f, y)
        return di_term + lam * bce

except ImportError:  # pragma: no cover
    pass


if __name__ == "__main__":
    # Demonstration only; the assertions live in tests/test_dial_objective.py.
    rng = np.random.default_rng(0)
    n, a, kappa, lam = 500, 0.7, 0.5, 0.5
    y = (rng.random(n) < 0.1).astype(float)
    f = rng.normal(size=n)

    class _TD:
        def get_label(self):
            return y

    grad, hess = lightgbm_dial_objective(a, kappa, lam)(f, _TD())
    corr = tau_gradient_correction(f, y, a, kappa, 1.0)
    print("mean |grad| =", np.abs(grad).mean(), " mean |tau-correction| =", np.abs(corr).mean(),
          " ratio =", np.abs(corr).mean() / np.abs(grad).mean())
    print("non-defaulter gradient now non-zero:", np.abs(grad[y == 0]).mean() > 0)
