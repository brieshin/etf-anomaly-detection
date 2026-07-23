"""
Data acquisition and preprocessing.

Corresponds to Chapter 4.2 (Data and Preprocessing) of the dissertation.
Downloads adjusted close prices for the nine-ETF panel, computes daily
log returns, splits chronologically, normalises, and builds sliding
windows for the LSTM Autoencoder.
"""

import numpy as np
import pandas as pd
import yfinance as yf

from config import TICKERS, START_DATE, END_DATE, TRAIN_FRAC, VAL_FRAC


def download_prices(start=START_DATE, end=END_DATE, tickers=TICKERS):
    """Download adjusted close prices for all tickers, aligned on common trading days."""
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if "Close" in raw else raw
    prices = prices[tickers]           # enforce fixed column order
    prices = prices.dropna(how="any")  # keep only days where all nine ETFs traded
    return prices


def compute_log_returns(prices):
    """Daily log returns: r_t = ln(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1)).dropna(how="any")


def chronological_split(returns, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    """Chronological (non-shuffled) train/val/test split — see Chapter 4.2 for rationale."""
    n = len(returns)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train = returns.iloc[:train_end]
    val = returns.iloc[train_end:val_end]
    test = returns.iloc[val_end:]
    return train, val, test


class ReturnNormaliser:
    """Z-score normalisation fitted only on the training split (avoids look-ahead bias)."""

    def fit(self, train_returns):
        self.mean_ = train_returns.mean()
        self.std_ = train_returns.std()
        return self

    def transform(self, returns):
        return (returns - self.mean_) / self.std_


def make_windows(returns, window_length):
    """Build overlapping sliding windows of shape (n_windows, window_length, n_assets)."""
    values = returns.values.astype(np.float32)
    n_steps, n_assets = values.shape
    n_windows = n_steps - window_length + 1
    if n_windows <= 0:
        raise ValueError(
            f"window_length ({window_length}) is longer than the available "
            f"series ({n_steps} rows). Shorten the window or provide more data."
        )
    windows = np.lib.stride_tricks.sliding_window_view(
        values, window_shape=(window_length, n_assets)
    )
    return windows[:, 0, :, :].copy()  # .copy(): stride_tricks output is non-writable


def load_and_prepare(window_length):
    """End-to-end data pipeline: download -> returns -> split -> normalise -> window.

    Returns a dict with prices, returns, the three raw/normalised splits,
    the fitted normaliser, and the three window arrays.
    """
    prices = download_prices()
    returns = compute_log_returns(prices)
    print(f"Loaded {len(returns)} trading days of log returns across {returns.shape[1]} ETFs.")

    train_r, val_r, test_r = chronological_split(returns)
    print(f"train: {len(train_r)} days | val: {len(val_r)} days | test: {len(test_r)} days")
    print(f"train range: {train_r.index[0].date()} to {train_r.index[-1].date()}")
    print(f"val range:   {val_r.index[0].date()} to {val_r.index[-1].date()}")
    print(f"test range:  {test_r.index[0].date()} to {test_r.index[-1].date()}")

    normaliser = ReturnNormaliser().fit(train_r)
    train_n = normaliser.transform(train_r)
    val_n = normaliser.transform(val_r)
    test_n = normaliser.transform(test_r)

    train_w = make_windows(train_n, window_length)
    val_w = make_windows(val_n, window_length)
    test_w = make_windows(test_n, window_length)
    print(f"window shapes — train: {train_w.shape}, val: {val_w.shape}, test: {test_w.shape}")

    return {
        "prices": prices,
        "returns": returns,
        "train_r": train_r, "val_r": val_r, "test_r": test_r,
        "train_n": train_n, "val_n": val_n, "test_n": test_n,
        "normaliser": normaliser,
        "train_w": train_w, "val_w": val_w, "test_w": test_w,
    }
