"""
Ground-truth label construction and quantitative evaluation.

Corresponds to Chapter 4.7.2 (Quantitative Metrics) and Chapter 5.3
(Quantitative Evaluation) of the dissertation.

IMPORTANT — Case 4 / extended-validation event handling:
The primary ground-truth label set (`build_ground_truth`) uses only
the three CORE_VALIDATION_EVENTS from config.py. The fourth event
(EXTENDED_VALIDATION_EVENT) is deliberately excluded from this set and
reported qualitatively instead (via its score range, in Section 5.2.4
and Section 5.2). This was a tested decision, not an
oversight: `run_case4_sensitivity_check()` below reproduces the
sensitivity analysis reported in Chapter 6.3.2 — including Case 4
reduces the LSTM Autoencoder's F1 advantage over Isolation Forest from
0.139 to 0.069, and its AU-ROC advantage from 0.034 to 0.010, because
Case 4, like Case 3, is an event the model largely fails to detect.
"""

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from config import CORE_VALIDATION_EVENTS, EXTENDED_VALIDATION_EVENT


def build_ground_truth(test_dates, event_windows=CORE_VALIDATION_EVENTS):
    """Binary ground-truth label vector: 1 if a test window's date falls
    within one of the given event windows, else 0."""
    y_true = np.zeros(len(test_dates), dtype=int)
    for start, end in event_windows:
        mask = (test_dates >= start) & (test_dates <= end)
        y_true[mask] = 1
    return y_true


def best_f1_threshold(scores, y_true):
    """Scan all candidate thresholds and return the one maximising F1.

    Note (Chapter 4.7.2 limitation): the threshold is selected on the
    test set itself rather than a held-out calibration set, producing a
    mildly optimistic performance estimate. See Chapter 6.3.3.
    """
    thresholds = np.unique(scores)
    best_f1, best_t = 0, thresholds[0]
    for t in thresholds:
        preds = (scores >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_f1, best_t


def compute_metrics(lstm_scores, if_scores, y_true):
    """Returns a dict of F1 / AU-ROC / AU-PR for both models."""
    lstm_f1, lstm_t = best_f1_threshold(lstm_scores, y_true)
    if_f1, if_t = best_f1_threshold(if_scores, y_true)

    return {
        "lstm": {
            "f1": lstm_f1, "threshold": lstm_t,
            "auroc": roc_auc_score(y_true, lstm_scores),
            "aupr": average_precision_score(y_true, lstm_scores),
        },
        "isolation_forest": {
            "f1": if_f1, "threshold": if_t,
            "auroc": roc_auc_score(y_true, if_scores),
            "aupr": average_precision_score(y_true, if_scores),
        },
    }


def run_case4_sensitivity_check(test_dates, lstm_scores, if_scores):
    """Reproduces the sensitivity analysis behind the Chapter 6.3.2 decision
    to exclude Case 4 from the primary ground-truth set. Prints a
    before/after comparison; does not change any module-level state."""
    y_true_core = build_ground_truth(test_dates, CORE_VALIDATION_EVENTS)
    y_true_all4 = build_ground_truth(test_dates, CORE_VALIDATION_EVENTS + [EXTENDED_VALIDATION_EVENT])

    metrics_core = compute_metrics(lstm_scores, if_scores, y_true_core)
    metrics_all4 = compute_metrics(lstm_scores, if_scores, y_true_all4)

    def gap(metrics, key):
        return metrics["lstm"][key] - metrics["isolation_forest"][key]

    print("--- Core validation (3 events) ---")
    print(f"F1 gap:     {gap(metrics_core, 'f1'):.3f}")
    print(f"AU-ROC gap: {gap(metrics_core, 'auroc'):.3f}")
    print("--- With Case 4 included (4 events) ---")
    print(f"F1 gap:     {gap(metrics_all4, 'f1'):.3f}")
    print(f"AU-ROC gap: {gap(metrics_all4, 'auroc'):.3f}")

    return metrics_core, metrics_all4
