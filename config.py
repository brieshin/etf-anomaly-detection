"""
Configuration constants for the ETF anomaly detection pipeline.

Corresponds to Chapter 3 (Data and Scope) and Chapter 4 (Methodology)
of the dissertation. Do not change TEST period boundaries or TICKERS
after results have been reported in the dissertation chapters, since
those values are cited directly in the text (e.g. peak Mahalanobis
score of 367.47 in Case 1).
"""

import torch

# --- ETF universe (Chapter 3.2) ---
TICKERS = ["SPY", "QQQ", "IWM", "TLT", "LQD", "HYG", "GLD", "USO", "DBC"]

ASSET_GROUPS = {
    "equities": ["SPY", "QQQ", "IWM"],
    "fixed_income": ["TLT", "LQD", "HYG"],
    "commodities": ["GLD", "USO", "DBC"],
}

# --- Data window (Chapter 4.2) ---
START_DATE = "2021-01-01"
END_DATE = "2026-07-01"  # fixed; do not extend, see module docstring above

TRAIN_FRAC = 0.6
VAL_FRAC = 0.2  # test gets the remaining 0.2

# --- Model hyperparameters (Chapter 4.3) ---
WINDOW_LENGTH = 20  # trading days per window (~1 month)
HIDDEN_SIZE = 32    # LSTM hidden size for encoder and decoder
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10        # early stopping patience, in epochs

# --- Reproducibility ---
SEED = 0

# --- Validated events (Chapter 3.3 / Chapter 5) ---
# Core-validation events: used to build the quantitative ground-truth
# label set (F1, AU-ROC, AU-PR) in Chapter 4.7.2 / Chapter 5.3.
CORE_VALIDATION_EVENTS = [
    ("2026-01-30", "2026-02-17"),  # Case 1: Feb 2026 selloff (intermetric)
    ("2026-03-04", "2026-04-20"),  # Case 2: Mar 2026 oil shock (trend)
    ("2026-06-05", "2026-06-10"),  # Case 3: Jun 2026 chip selloff (magnitude-driven)
]

# Extended-validation event: reported qualitatively only (Chapter 4.7.1,
# Chapter 6.3.2). NOT included in CORE_VALIDATION_EVENTS by default —
# see evaluate.py's `run_case4_sensitivity_check()` for the tested
# alternative and why it was not adopted as the primary ground truth.
EXTENDED_VALIDATION_EVENT = ("2026-05-12", "2026-05-22")  # Case 4: May 2026 bond yield surge (shapelet)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
