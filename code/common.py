"""Shared helpers for the experiment scripts.

Every script is deterministic given the seeds set in it or listed below.
Simulator-based quantities are labelled `source="simulator"` in every output file.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

# The repository root is found once, from SME_ATTAIN_ROOT if it is set and otherwise from this
# file's own location, and every path below is built from it.
ROOT = os.path.abspath(os.environ.get("SME_ATTAIN_ROOT",
                                      os.path.join(os.path.dirname(__file__), "..")))
LIB = os.path.join(ROOT, "lib")
if LIB not in sys.path:
    sys.path.insert(0, LIB)

RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")
DATA = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")
os.makedirs(RESULTS, exist_ok=True)

APPROVAL_RATES = [0.5, 0.6, 0.7, 0.8, 0.9]
A_OP = 0.7
SEEDS = [0, 1, 2, 3, 4]

# Sample sizes / base rates the paper reports for the two large real datasets.
# Used ONLY to parameterise the simulator's estimator-validation cells (Supplementary S4).
SBA_N, SBA_BASE = 899_164, 0.18
HC_N, HC_BASE = 307_511, 0.081


def dump(name: str, obj) -> str:
    path = os.path.join(RESULTS, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=_default)
    print("wrote", path, flush=True)
    return path


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# Controlled DGP with a KNOWN conditional default probability eta_S(x).
# This is the simulation study of Supplementary S4/S5: features are observed,
# an unobserved component of variance `noise` is integrated out analytically by
# Gauss-Hermite quadrature, so eta_S is exact.
# ---------------------------------------------------------------------------
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(31)
_GH_W = _GH_W / _GH_W.sum()


def known_eta_dgp(n: int, d: int, base_rate: float, noise: float, seed: int):
    """Return (X, y, eta) with eta = P(y=1|X) exact.

    X ~ N(0, I_d); linear index mu = X @ beta; latent z = mu + noise*e, e ~ N(0,1);
    y ~ Bernoulli(sigmoid(z + c)) with c solved so that E[eta] = base_rate.
    eta(x) = E_e[sigmoid(mu + noise*e + c)] by Gauss-Hermite quadrature.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    beta = rng.standard_normal(d)
    beta = beta / np.linalg.norm(beta) * 1.5  # fixed signal-to-noise scale
    mu = X @ beta

    def eta_of(c):
        return (sigmoid(mu[:, None] + noise * _GH_X[None, :] + c) * _GH_W[None, :]).sum(1)

    lo, hi = -20.0, 20.0
    for _ in range(60):
        c = 0.5 * (lo + hi)
        if eta_of(c).mean() > base_rate:
            hi = c
        else:
            lo = c
    c = 0.5 * (lo + hi)
    eta = eta_of(c)
    y = (rng.random(n) < eta).astype(float)
    return X, y, eta


def true_ceiling(eta: np.ndarray, a: float) -> float:
    """DI*(S,a) = mean of the lowest a-fraction of eta."""
    k = max(1, int(round(a * len(eta))))
    return float(np.sort(eta)[:k].mean())
