"""Central configuration for the privacy-utility experiments.

Every number that influences a result lives here so the whole study can be
audited (and changed) in one place. Hyper-parameters were fixed *before*
looking at private-model results and are NOT tuned per epsilon (tuning on the
private data would itself consume privacy budget; see paper Sec. 5).
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_ADULT = DATA_DIR / "raw" / "adult.csv"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

# Public mirror of the UCI Adult data (train + test concatenated, 48,842 rows).
# The original UCI host is used as a second option. The SHA-256 below pins the
# exact file the committed results were produced from.
ADULT_URLS = [
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
]
ADULT_SHA256 = "21e0ea2f925a00338929a8c86c27354c72ac1c79819bcca81c7d91c3d64218c2"

# --------------------------------------------------------------------------- #
# Experimental protocol
# --------------------------------------------------------------------------- #
TEST_SIZE = 0.30          # stratified hold-out fraction
SPLIT_SEED = 0            # one fixed train/test split for every model
SEEDS = list(range(10))   # repetitions for every randomised (private) model

# Privacy budgets to sweep (log-spaced, from "very strong" to "very weak").
EPSILONS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

# delta for the (epsilon, delta)-DP algorithm (DP-SGD). Rule of thumb: delta
# much smaller than 1/n; n_train ~ 34k here, so 1e-5 is ~1/3 of 1/n.
DELTA = 1e-5

# --------------------------------------------------------------------------- #
# Model hyper-parameters (shared across all epsilons)
# --------------------------------------------------------------------------- #
# Non-private and objective-perturbation logistic regression.
LOGREG_C = 1.0            # inverse L2-regularisation strength
LOGREG_MAX_ITER = 1000

# DP-SGD (our NumPy implementation, see src/dp_sgd.py).
DPSGD = dict(
    epochs=10,            # passes over the data (in expectation)
    batch_size=256,       # *expected* lot size L under Poisson sampling
    lr=0.5,               # constant learning rate
    clip_norm=1.0,        # per-example gradient L2 clipping bound C
    l2=1e-4,              # weight decay (data-independent, no privacy cost)
)

# Secondary experiment: how does training-set size change the trade-off?
SIZE_SWEEP_N = [1000, 3000, 10000, 34189]   # last value = full training set
SIZE_SWEEP_EPSILONS = [0.1, 1.0]
SIZE_SWEEP_SEEDS = list(range(5))
