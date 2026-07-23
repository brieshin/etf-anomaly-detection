"""
End-to-end pipeline: data -> train -> score -> evaluate -> visualise.

Run with: python main.py

Reproduces the dissertation's headline numbers (with SEED=0 and the
config.py defaults unchanged):
    - LSTM-AE Mahalanobis peak score: 367.47 (Case 1)
    - Training: early stopping at epoch 44, best val_loss ~0.798
    - Core-validation metrics: LSTM-AE F1=0.743, AU-ROC=0.913, AU-PR=0.675
                                Isolation Forest F1=0.604, AU-ROC=0.879, AU-PR=0.658
"""

import numpy as np
import torch

from config import TICKERS, WINDOW_LENGTH, SEED, DEVICE, CORE_VALIDATION_EVENTS
import data
import train as train_module
import scoring
import evaluate
import visualize
from model import extract_latents


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print("Using device:", DEVICE)

    # 1. Data
    d = data.load_and_prepare(WINDOW_LENGTH)

    # 2. Train
    model, history = train_module.train_model(d["train_w"], d["val_w"], n_features=len(TICKERS))
    visualize.plot_training_curves(history)

    # 3. Reconstruction errors and Mahalanobis scoring
    val_errors = train_module.reconstruction_errors(model, d["val_w"])
    test_errors = train_module.reconstruction_errors(model, d["test_w"])
    scorer = scoring.MahalanobisScorer().fit(val_errors)
    lstm_scores = scorer.score(test_errors, reduction="max")
    print("LSTM-AE Mahalanobis scores — min:", lstm_scores.min(), "max:", lstm_scores.max())

    # 4. Isolation Forest baseline
    if_scores = scoring.isolation_forest_baseline(d["train_w"], d["test_w"])
    print("Isolation Forest scores — min:", if_scores.min(), "max:", if_scores.max())

    test_dates = d["test_n"].index[WINDOW_LENGTH - 1:]
    visualize.plot_score_comparison(test_dates, lstm_scores, if_scores)

    # 5. Ground truth and quantitative evaluation (core-validation events only)
    y_true = evaluate.build_ground_truth(test_dates, CORE_VALIDATION_EVENTS)
    metrics = evaluate.compute_metrics(lstm_scores, if_scores, y_true)
    print("LSTM-AE:", metrics["lstm"])
    print("Isolation Forest:", metrics["isolation_forest"])
    visualize.plot_precision_recall(
        y_true, lstm_scores, if_scores,
        metrics["lstm"]["aupr"], metrics["isolation_forest"]["aupr"],
    )

    # 6. Latent space visualisation
    test_latents = extract_latents(model, d["test_w"], DEVICE)
    visualize.plot_latent_space(test_latents, y_true, lstm_scores)

    # 7. Case 4 sensitivity check (see Chapter 6.3.2 for why Case 4 is
    #    excluded from the primary ground-truth set)
    evaluate.run_case4_sensitivity_check(test_dates, lstm_scores, if_scores)

    return model, d, lstm_scores, if_scores, y_true, metrics


if __name__ == "__main__":
    main()
