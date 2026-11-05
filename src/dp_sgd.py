"""DP-SGD (Abadi et al., 2016) for logistic regression, written in plain NumPy.

Algorithm (one step t = 1..T):
  1. Poisson-sample a lot B_t: each example joins independently w.p. q = L/n.
  2. Per-example gradient g_i of the logistic loss (closed form for a linear model).
  3. Clip:   g_i <- g_i / max(1, ||g_i||_2 / C)          (bounds sensitivity to C)
  4. Noise:  g~  = (sum_i g_i + N(0, sigma^2 C^2 I)) / L  (L = *expected* lot size)
  5. Update: theta <- theta - lr * (g~ + l2 * theta)
The noise multiplier sigma is calibrated with our RDP accountant so the whole
run satisfies (epsilon, delta)-DP w.r.t. add/remove-one-record adjacency.

Dividing by the fixed expected lot size L (not the realised |B_t|) matters: the
realised size is itself private, so using it would break the analysis.
"""
from __future__ import annotations

import numpy as np

from .rdp_accountant import calibrate_sigma, compute_epsilon


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))        # numerically stable form


class DPSGDLogisticRegression:
    """Binary logistic regression trained with DP-SGD.

    Parameters
    ----------
    epsilon : float or None
        Target privacy budget. ``None`` trains the *same* algorithm with no noise
        and no clipping, i.e. ordinary mini-batch SGD (a matched non-private
        reference isolating the cost of privacy from the cost of the optimiser).
    delta, epochs, batch_size, lr, clip_norm, l2, random_state : see config.DPSGD.
    """

    def __init__(self, epsilon=1.0, delta=1e-5, epochs=10, batch_size=256, lr=0.5,
                 clip_norm=1.0, l2=1e-4, random_state=None):
        self.epsilon, self.delta = epsilon, delta
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.clip_norm, self.l2 = clip_norm, l2
        self.random_state = random_state

    # The training schedule is a function of public quantities (n, L, epochs)
    # only, so it can be fixed before touching the data.
    def _schedule(self, n: int):
        q = min(1.0, self.batch_size / n)
        steps = int(np.ceil(self.epochs / q))
        return q, steps

    def fit(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape
        Xb = np.hstack([X, np.ones((n, 1))])       # append bias column
        q, steps = self._schedule(n)
        L = q * n
        private = self.epsilon is not None
        if private and np.isinf(self.epsilon):
            self.sigma_, self.spent_epsilon_ = 0.0, np.inf      # clip-only ablation
        elif private:
            self.sigma_ = calibrate_sigma(self.epsilon, q, steps, self.delta)
            self.spent_epsilon_ = compute_epsilon(q, self.sigma_, steps, self.delta)
        else:
            self.sigma_, self.spent_epsilon_ = 0.0, np.inf

        theta = np.zeros(d + 1)
        for _ in range(steps):
            idx = np.flatnonzero(rng.random(n) < q)          # Poisson sampling
            xb, yb = Xb[idx], y[idx]
            # Per-example gradient of log-loss: (sigmoid(x.theta) - y) * x
            per_ex = (_sigmoid(xb @ theta) - yb)[:, None] * xb
            if private:
                norms = np.linalg.norm(per_ex, axis=1, keepdims=True)
                per_ex = per_ex / np.maximum(1.0, norms / self.clip_norm)
                noise = (rng.normal(0.0, self.sigma_ * self.clip_norm, size=d + 1)
                         if self.sigma_ > 0 else 0.0)
                grad = (per_ex.sum(axis=0) + noise) / L
            else:
                grad = per_ex.sum(axis=0) / max(len(idx), 1)
            theta -= self.lr * (grad + self.l2 * theta)

        self.coef_, self.intercept_ = theta[:-1], theta[-1]
        self.n_steps_, self.sample_rate_ = steps, q
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = _sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) > 0).astype(int)
