"""Renyi-DP accountant for DP-SGD (Poisson-subsampled Gaussian mechanism).

Implemented from first principles (no external DP library) so that the DP-SGD
baseline is fully transparent.

Background
----------
One DP-SGD step samples each training example independently with probability
``q`` (Poisson sampling), sums clipped per-example gradients (L2 sensitivity
``C``) and adds Gaussian noise ``N(0, (sigma*C)^2 I)``. For integer Renyi
orders ``alpha >= 2`` the RDP of this *sampled Gaussian mechanism* is exactly
(Mironov, Talwar & Zhang, 2019, Sec. 3.3):

    eps_RDP(alpha) = 1/(alpha-1) * log( sum_{k=0}^{alpha} binom(alpha, k)
                         * (1-q)^(alpha-k) * q^k * exp((k^2 - k) / (2 sigma^2)) )

RDP composes additively over the ``T`` steps, and is converted to
(eps, delta)-DP with the tighter conversion of Balle et al. (2020, Thm. 21):

    eps = min_alpha  T*eps_RDP(alpha) + log((alpha-1)/alpha)
                     - (log(delta) + log(alpha)) / (alpha - 1)
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln, logsumexp

# Integer Renyi orders to optimise over (dense for small, sparse for large).
ORDERS = list(range(2, 65)) + [80, 96, 128, 160, 192, 256, 384, 512, 1024, 2048, 4096]
# Note: the largest order sets the smallest certifiable epsilon (about
# -log(delta)/alpha_max), ~0.0007 at delta = 1e-5 with alpha_max = 4096.


def _log_binom(n: int, k: np.ndarray) -> np.ndarray:
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def rdp_sampled_gaussian(q: float, sigma: float, alpha: int) -> float:
    """RDP of ONE step of the Poisson-subsampled Gaussian mechanism at order alpha."""
    if q == 0:
        return 0.0
    if sigma == 0:
        return math.inf
    if q == 1.0:                       # no subsampling: plain Gaussian mechanism
        return alpha / (2 * sigma ** 2)
    k = np.arange(alpha + 1)
    log_terms = (_log_binom(alpha, k) + (alpha - k) * math.log1p(-q)
                 + k * math.log(q) + (k * k - k) / (2 * sigma ** 2))
    return float(logsumexp(log_terms) / (alpha - 1))


def compute_epsilon(q: float, sigma: float, steps: int, delta: float) -> float:
    """(eps, delta)-DP guarantee of `steps` DP-SGD iterations."""
    best = math.inf
    for a in ORDERS:
        rdp = steps * rdp_sampled_gaussian(q, sigma, a)
        eps = rdp + math.log((a - 1) / a) - (math.log(delta) + math.log(a)) / (a - 1)
        best = min(best, eps)
    return max(best, 0.0)


def calibrate_sigma(target_eps: float, q: float, steps: int, delta: float,
                    lo: float = 0.05, hi: float = 2000.0, tol: float = 1e-3) -> float:
    """Smallest noise multiplier sigma s.t. the run is (target_eps, delta)-DP.

    eps is monotonically decreasing in sigma, so a bisection in log-space works.
    """
    while compute_epsilon(q, hi, steps, delta) > target_eps:   # grow upper bracket
        hi *= 4
        if hi > 1e9:
            raise ValueError("target epsilon unreachable")
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        if compute_epsilon(q, mid, steps, delta) > target_eps:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1 + tol:
            break
    return hi        # conservative side: guaranteed eps <= target
