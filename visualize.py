"""
Plotting utilities.

Corresponds to figures used in Chapter 5 (Results) and the latent
space visualisation underlying Chapter 6.2.1's macro-attribution
discussion.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve


def plot_returns(returns):
    returns.plot(figsize=(12, 4), legend=True, title="Daily log returns, all 9 ETFs")
    plt.tight_layout()
    plt.show()


def plot_training_curves(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("MSE reconstruction loss")
    plt.legend()
    plt.title("Training curves")
    plt.tight_layout()
    plt.show()


def plot_score_comparison(test_dates, lstm_scores, if_scores):
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(test_dates, lstm_scores, color="tab:red")
    axes[0].set_title("LSTM Autoencoder — Mahalanobis anomaly score")
    axes[1].plot(test_dates, if_scores, color="tab:blue")
    axes[1].set_title("Isolation Forest — anomaly score")
    plt.tight_layout()
    plt.show()


def plot_precision_recall(y_true, lstm_scores, if_scores, lstm_aupr, if_aupr):
    precision_l, recall_l, _ = precision_recall_curve(y_true, lstm_scores)
    precision_i, recall_i, _ = precision_recall_curve(y_true, if_scores)

    plt.figure(figsize=(7, 5))
    plt.plot(recall_l, precision_l, label=f"LSTM-AE (AU-PR={lstm_aupr:.3f})")
    plt.plot(recall_i, precision_i, label=f"Isolation Forest (AU-PR={if_aupr:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.show()


def plot_latent_space(test_latents, y_true, lstm_scores):
    """Reproduces the two-panel latent space figure underlying Chapter
    6.2.1 (labelled anomaly windows vs. continuous anomaly score)."""
    pca = PCA(n_components=2, random_state=0)
    latents_2d = pca.fit_transform(test_latents)
    explained = pca.explained_variance_ratio_
    print(f"PCA explained variance — PC1: {explained[0]*100:.1f}%, PC2: {explained[1]*100:.1f}% "
          f"(total: {explained.sum()*100:.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    colors = np.where(y_true == 1, "tab:red", "tab:blue")
    axes[0].scatter(latents_2d[:, 0], latents_2d[:, 1], c=colors, s=18, alpha=0.7, edgecolors="none")
    axes[0].set_title("Latent space — labelled anomaly windows")
    axes[0].set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", markersize=8, label="Normal"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:red", markersize=8, label="Labelled anomaly"),
    ]
    axes[0].legend(handles=handles, loc="best")

    sc = axes[1].scatter(latents_2d[:, 0], latents_2d[:, 1], c=lstm_scores, cmap="viridis", s=18, alpha=0.8, edgecolors="none")
    axes[1].set_title("Latent space — Mahalanobis anomaly score")
    axes[1].set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    plt.colorbar(sc, ax=axes[1], label="anomaly score")

    plt.tight_layout()
    plt.show()

    return latents_2d, explained
