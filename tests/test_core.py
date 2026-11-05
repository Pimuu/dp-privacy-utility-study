"""Sanity tests for the privacy machinery and the data pipeline.

Run with either
    python -m tests.test_core        (no extra dependency)
    python -m pytest tests           (if pytest is installed)
"""
import math

import numpy as np

from src import config
from src.data import CATEGORIES, NUMERIC_BOUNDS, load_dataset, make_synthetic, max_row_norm
from src.dp_sgd import DPSGDLogisticRegression
from src.rdp_accountant import (ORDERS, calibrate_sigma, compute_epsilon,
                                rdp_sampled_gaussian)


# ----------------------------- accountant ---------------------------------- #
def test_accountant_matches_tensorflow_privacy_reference():
    """TF-Privacy's MNIST tutorial (n=60000, L=256, sigma=1.1, 60 epochs, delta=1e-5)
    reports eps ~= 3.01 with the classic RDP->DP conversion. Reproduce it."""
    q = 256 / 60000
    steps = int(60 / q)
    classic = min(steps * rdp_sampled_gaussian(q, 1.1, a) - math.log(1e-5) / (a - 1)
                  for a in ORDERS)
    assert abs(classic - 3.01) < 0.02, classic
    # The tighter conversion we actually use must never be looser.
    assert compute_epsilon(q, 1.1, steps, 1e-5) <= classic


def test_full_batch_is_plain_gaussian_rdp():
    """q = 1: RDP of the Gaussian mechanism is alpha / (2 sigma^2)."""
    for a in (2, 5, 32):
        assert math.isclose(rdp_sampled_gaussian(1.0, 2.0, a), a / 8)


def test_subsampling_amplifies_privacy():
    assert rdp_sampled_gaussian(0.01, 1.0, 8) < rdp_sampled_gaussian(1.0, 1.0, 8)


def test_epsilon_decreases_with_noise_and_calibration_is_conservative():
    q, T, d = 0.01, 1000, 1e-5
    eps = [compute_epsilon(q, s, T, d) for s in (0.6, 1.0, 2.0, 5.0)]
    assert all(a > b for a, b in zip(eps, eps[1:]))
    for target in (0.1, 1.0, 8.0):
        sigma = calibrate_sigma(target, q, T, d)
        assert compute_epsilon(q, sigma, T, d) <= target
        assert compute_epsilon(q, sigma * 0.99, T, d) > target * 0.98  # near-tight


# ------------------------------- DP-SGD ------------------------------------ #
def test_dpsgd_spends_at_most_its_budget_and_is_deterministic():
    X, y, _, _ = make_synthetic(n=3000, d=6, seed=1)
    m1 = DPSGDLogisticRegression(epsilon=1.0, epochs=3, random_state=7).fit(X, y)
    m2 = DPSGDLogisticRegression(epsilon=1.0, epochs=3, random_state=7).fit(X, y)
    assert m1.spent_epsilon_ <= 1.0
    assert np.allclose(m1.coef_, m2.coef_)


def test_dpsgd_learns_and_noise_hurts():
    X, y, _, _ = make_synthetic(n=6000, seed=0)
    acc = lambda m: (m.predict(X) == y).mean()  # noqa: E731
    nonpriv = acc(DPSGDLogisticRegression(epsilon=None, epochs=5, random_state=0).fit(X, y))
    tiny = acc(DPSGDLogisticRegression(epsilon=0.005, epochs=5, random_state=0).fit(X, y))
    assert nonpriv > max(y.mean(), 1 - y.mean()) + 0.05   # beats majority class
    assert tiny < nonpriv


# -------------------------------- data ------------------------------------- #
def test_public_norm_bound_holds_and_schema_is_fixed():
    ds = load_dataset()
    expected_dim = len(NUMERIC_BOUNDS) + sum(len(v) + 1 for v in CATEGORIES.values())
    if ds.source == "adult":
        assert ds.X_train.shape[1] == expected_dim
        assert math.isclose(ds.max_row_norm, max_row_norm())
    for X in (ds.X_train, ds.X_test):
        assert X.min() >= 0 and X.max() <= 1
        assert np.linalg.norm(X, axis=1).max() <= ds.max_row_norm + 1e-9
    n = len(ds.y_train) + len(ds.y_test)
    assert abs(len(ds.y_test) / n - config.TEST_SIZE) < 0.01


if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")
