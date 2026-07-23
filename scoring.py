"""
Anomaly scoring: the Mahalanobis distance scorer (Malhotra et al., 2016)
and the Isolation Forest baseline.

Corresponds to Chapter 4.5 (Anomaly Scoring) and Chapter 4.6 (Baseline
Model) of the dissertation.

Note on dimensionality (a non-obvious fix worth preserving): the
Gaussian is fitted in the low-dimensional n_features=9 space, treating
every individual timestep across every validation window as one
sample, NOT by flattening entire windows to window_length * n_features
= 180 dimensions. The latter makes the covariance matrix ill-conditioned
given only ~256 validation windows.
"""

import numpy as np
from sklearn.ensemble import IsolationForest


class MahalanobisScorer:
    """Fits a multivariate Gaussian over validation reconstruction errors."""

    def fit(self, val_errors):
        points = val_errors.reshape(-1, val_errors.shape[-1])
        self.mu_ = points.mean(axis=0)
        cov = np.cov(points, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6  # numerical stability
        self.cov_inv_ = np.linalg.inv(cov)
        return self

    def score_points(self, errors):
        points = errors.reshape(-1, errors.shape[-1])
        diff = points - self.mu_
        point_scores = np.einsum("ij,jk,ik->i", diff, self.cov_inv_, diff)
        return point_scores.reshape(errors.shape[0], errors.shape[1])

    def score(self, errors, reduction="max"):
        """reduction='max' is the primary scoring rule used throughout the
        dissertation. reduction='mean' is used only as a robustness check
        in Chapter 5, Case 2 (Section 5.2.2)."""
        point_scores = self.score_points(errors)
        if reduction == "max":
            return point_scores.max(axis=1)
        elif reduction == "mean":
            return point_scores.mean(axis=1)
        raise ValueError("reduction must be 'max' or 'mean'")


def isolation_forest_baseline(train_windows, test_windows, random_state=42):
    """Isolation Forest baseline, fit on flattened training windows."""
    train_flat = train_windows.reshape(train_windows.shape[0], -1)
    test_flat = test_windows.reshape(test_windows.shape[0], -1)
    clf = IsolationForest(random_state=random_state, contamination="auto")
    clf.fit(train_flat)
    return -clf.score_samples(test_flat)  # higher = more anomalous
