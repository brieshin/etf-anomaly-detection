# ETF Anomaly Detection — LSTM Autoencoder

Code accompanying the dissertation *"Anomaly Detection in ETF Markets Using
LSTM Autoencoder: A Multi-Asset Group Risk Analysis"*.

A jointly trained LSTM Autoencoder (EncDec-AD, following Malhotra et al.,
2016) for anomaly detection across nine ETFs spanning three asset groups,
benchmarked against an Isolation Forest baseline and validated against four
independently confirmed historical market events.

## Repository structure

| File | Purpose | Dissertation reference |
|---|---|---|
| `config.py` | All constants: ETF universe, date range, hyperparameters, validated event windows | Chapter 3, Chapter 4.3 |
| `data.py` | Download prices, compute log returns, chronological split, normalisation, sliding windows | Chapter 4.2 |
| `model.py` | `EncDecAD` architecture (LSTM encoder/decoder) and latent-vector extraction | Chapter 4.3 |
| `train.py` | Training loop with early stopping; reconstruction error computation | Chapter 4.3, Chapter 5.1 |
| `scoring.py` | `MahalanobisScorer` and the Isolation Forest baseline | Chapter 4.5, Chapter 4.6 |
| `evaluate.py` | Ground-truth label construction, F1/AU-ROC/AU-PR, the Case 4 sensitivity check | Chapter 4.7.2, Chapter 5.3, Chapter 6.3.2 |
| `visualize.py` | Training curves, score comparison, PR curve, latent space plots | Chapter 5, Chapter 6.2.1 |
| `event_inspect.py` | Ad-hoc helpers for inspecting returns/scores around a specific date window | Chapter 5.2 |
| `main.py` | End-to-end pipeline entry point | — |

## Setup

```bash
pip install -r requirements.txt
```

Requires network access to Yahoo Finance (`query1.finance.yahoo.com`,
`query2.finance.yahoo.com`) to download price data. If running in a
sandboxed environment, confirm this domain is on the network allowlist
before running `main.py`.

## Running the full pipeline

```bash
python main.py
```

This downloads data, trains the model, computes anomaly scores for both
models, evaluates against the core-validation ground truth, and produces
all figures used in Chapters 5 and 6.

## Reproducibility

With `config.py` unchanged (`SEED = 0`, `END_DATE = "2026-07-01"`), the
pipeline reproduces the dissertation's headline numbers:

- **Training**: early stopping at epoch 44, best validation loss ≈ 0.798
- **LSTM-AE Mahalanobis peak score**: 367.47 (Case 1, the February 2026
  cross-asset selloff)
- **Core-validation quantitative metrics** (Cases 1–3 only, see below):

  | Model | F1 | AU-ROC | AU-PR |
  |---|---|---|---|
  | LSTM Autoencoder | 0.743 | 0.913 | 0.675 |
  | Isolation Forest | 0.604 | 0.879 | 0.658 |

Do not change `END_DATE` or `TICKERS` without expecting these numbers to
shift — the train/val/test date boundaries move whenever the underlying
Yahoo Finance data series is extended, which changes early-stopping
behaviour and, in turn, all downstream scores. This was discovered
directly during this project: an unintentional 10-day data extension
changed the peak Mahalanobis score from 367.47 to 307.13.

## Why Case 4 is excluded from the primary ground truth

Four historical events were validated in this dissertation, spanning all
four categories of this dissertation's anomaly typology (three drawn
directly from Darban et al. (2024): intermetric, trend, and shapelet;
a fourth, structure-preserving magnitude, added after being observed
empirically in the results). Only the
first three (Cases 1–3) are used to build the ground-truth label set for
F1/AU-ROC/AU-PR. Case 4 (the mid-May 2026 bond yield surge) is reported
qualitatively only, via its score range.

This was a tested decision, not an oversight. `evaluate.run_case4_sensitivity_check()`
reproduces the sensitivity analysis: including Case 4 in the ground truth
reduces the LSTM Autoencoder's F1 advantage over Isolation Forest from
0.139 to 0.069, and its AU-ROC advantage from 0.034 to 0.010, because Case
4, like Case 3 is an event the model largely fails to detect. See
Chapter 6.3.2 of the dissertation for the full discussion.

## Key finding

The model detects breaks in learned cross-asset correlation structure
sharply and specifically (Case 1), but is comparatively insensitive to
large-magnitude, trend-type, or shapelet-type anomalies that preserve
normal cross-asset correlation (Cases 2–4), regardless of their economic
significance. See Chapter 6.2 and 6.2.1 for the full mechanistic and
macro-economic account.
