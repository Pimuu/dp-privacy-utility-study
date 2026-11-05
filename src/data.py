"""Data loading and *privacy-aware* preprocessing.

Why preprocessing matters for differential privacy
--------------------------------------------------
A DP guarantee covers only what the algorithm computes from the private data.
If we standardised features with the empirical mean/std, or picked one-hot
categories from the observed values, those statistics would leak information
outside the privacy accounting. We therefore build the feature map from a
**public schema** only:

* numeric features are clipped to fixed, documented domain bounds and min-max
  scaled with those bounds (never with data-derived min/max);
* categorical features are one-hot encoded against the fixed vocabularies from
  the dataset documentation (UCI ``adult.names``), plus an ``Unknown`` slot.

A useful by-product: every encoded row has a *known* maximum L2 norm
(``max_row_norm``), which the objective-perturbation mechanism needs as
``data_norm``. No data-dependent quantity enters any private model.
"""
from __future__ import annotations

import hashlib
import urllib.request
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]

# Public domain bounds (from the UCI documentation / census coding), NOT from data.
# Heavy-tailed money features are log1p-transformed before scaling.
NUMERIC_BOUNDS = {
    "age": (17, 90, False),
    "education_num": (1, 16, False),
    "hours_per_week": (1, 99, False),
    "capital_gain": (0, 99999, True),   # (low, high, log1p?)
    "capital_loss": (0, 4356, True),
}

# Fixed vocabularies from UCI adult.names. Missing ('?') or unseen -> 'Unknown'.
CATEGORIES = {
    "workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
                  "Local-gov", "State-gov", "Without-pay", "Never-worked"],
    "marital_status": ["Married-civ-spouse", "Divorced", "Never-married", "Separated",
                       "Widowed", "Married-spouse-absent", "Married-AF-spouse"],
    "occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales",
                   "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
                   "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
                   "Transport-moving", "Priv-house-serv", "Protective-serv",
                   "Armed-Forces"],
    "relationship": ["Wife", "Own-child", "Husband", "Not-in-family",
                     "Other-relative", "Unmarried"],
    "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
    "sex": ["Female", "Male"],
    "native_country": [
        "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
        "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China",
        "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica",
        "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic",
        "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala",
        "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador",
        "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"],
}
# Dropped on purpose: 'fnlwgt' is a census sampling weight, not a property of
# the person; 'education' is a 1:1 duplicate of 'education_num'.


@dataclass
class Dataset:
    """Encoded train/test split plus the metadata the private models need."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    max_row_norm: float      # public upper bound on ||x||_2 for any encoded row
    source: str              # "adult" or "synthetic"


# --------------------------------------------------------------------------- #
# Raw data acquisition
# --------------------------------------------------------------------------- #
def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_adult() -> bool:
    """Fetch the Adult CSV into data/raw/ if missing. Returns True on success."""
    target = config.RAW_ADULT
    if target.exists():
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    for url in config.ADULT_URLS:
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                target.write_bytes(resp.read())
            return True
        except Exception as exc:  # network blocked, 404, ...
            warnings.warn(f"Could not download {url}: {exc}")
    return False


def load_adult_raw() -> pd.DataFrame:
    """Read the raw Adult CSV (downloading it on first use)."""
    if not download_adult():
        raise FileNotFoundError("Adult dataset unavailable (offline?)")
    if _sha256(config.RAW_ADULT) != config.ADULT_SHA256:
        warnings.warn("adult.csv checksum differs from the pinned file; "
                      "numbers may not match the committed results exactly.")
    df = pd.read_csv(config.RAW_ADULT, header=None, names=COLUMNS,
                     skipinitialspace=True, na_values="?", comment="|")
    df = df.dropna(subset=["income"])
    # The UCI test file writes labels as '>50K.'; normalise both spellings.
    df["income"] = df["income"].str.strip().str.rstrip(".")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Public-schema feature map
# --------------------------------------------------------------------------- #
def encode(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Map raw records to a bounded numeric matrix using only public metadata."""
    blocks, names = [], []
    for col, (lo, hi, use_log) in NUMERIC_BOUNDS.items():
        v = df[col].astype(float).clip(lo, hi).to_numpy()
        if use_log:
            v, lo, hi = np.log1p(v), np.log1p(lo), np.log1p(hi)
        blocks.append(((v - lo) / (hi - lo))[:, None])       # in [0, 1]
        names.append(col)
    for col, vocab in CATEGORIES.items():
        full_vocab = vocab + ["Unknown"]
        values = df[col].where(df[col].isin(vocab), "Unknown")
        onehot = (values.to_numpy()[:, None] == np.array(full_vocab)[None, :])
        blocks.append(onehot.astype(float))
        names += [f"{col}={c}" for c in full_vocab]
    return np.hstack(blocks), names


def max_row_norm() -> float:
    """Data-independent bound: each numeric feature <= 1, one active slot per group."""
    return float(np.sqrt(len(NUMERIC_BOUNDS) + len(CATEGORIES)))


# --------------------------------------------------------------------------- #
# Synthetic fallback (used only if Adult cannot be obtained)
# --------------------------------------------------------------------------- #
def make_synthetic(n: int = 40000, d: int = 12, seed: int = 0):
    """Bounded tabular data with a known logistic ground truth.

    Features are uniform in [0, 1] (so the same norm bound logic applies) and the
    label is drawn from a sparse logistic model, giving a clear but noisy signal.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, d))
    w = np.zeros(d)
    w[: d // 2] = rng.normal(0, 3, size=d // 2)
    logits = (X - 0.5) @ w - 1.0
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logits))).astype(int)
    return X, y, [f"x{i}" for i in range(d)], float(np.sqrt(d))


def load_dataset(force_synthetic: bool = False) -> Dataset:
    """Return the encoded, split dataset (Adult if possible, else synthetic)."""
    source = "adult"
    if not force_synthetic:
        try:
            df = load_adult_raw()
            X, names = encode(df)
            y = (df["income"] == ">50K").astype(int).to_numpy()
            bound = max_row_norm()
        except FileNotFoundError as exc:
            warnings.warn(f"{exc}; falling back to the synthetic dataset.")
            force_synthetic = True
    if force_synthetic:
        source = "synthetic"
        X, y, names, bound = make_synthetic()

    # Sanity check: the public bound really holds for every row.
    assert np.linalg.norm(X, axis=1).max() <= bound + 1e-9
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.SPLIT_SEED)
    return Dataset(X_tr, X_te, y_tr, y_te, names, bound, source)
