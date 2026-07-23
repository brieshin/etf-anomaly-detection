"""
Ad-hoc helpers for inspecting returns and scores around a specific
date window. Used to produce the day-by-day tables in Chapter 5.2.
"""

import numpy as np


def inspect_returns(returns, start, end, tickers=None):
    """Print raw returns and cumulative return over [start, end] for the
    given tickers (or all tickers if None)."""
    window = returns.loc[start:end]
    if tickers:
        window = window[tickers]
    print(window)
    print(f"\nCumulative return per ticker ({start} to {end}):")
    print((np.exp(window.sum()) - 1).round(4) * 100)
    return window


def inspect_scores(test_dates, scores, start, end):
    """Print the anomaly score for each test date within [start, end]."""
    mask = (test_dates >= start) & (test_dates <= end)
    for d, s in zip(test_dates[mask], scores[mask]):
        print(f"{d.date()}  score={s:.2f}")
    return scores[mask]


def find_peak(test_dates, scores, context_days=10):
    """Report the peak score, its date, and scores in a window around it."""
    peak_idx = scores.argmax()
    peak_date = test_dates[peak_idx]
    print(f"Peak score: {scores[peak_idx]:.2f}, date: {peak_date.date()}")

    start = max(0, peak_idx - context_days)
    end = min(len(scores), peak_idx + context_days)
    for d, s in zip(test_dates[start:end], scores[start:end]):
        print(f"{d.date()}  score={s:.2f}")
    return peak_idx, peak_date
