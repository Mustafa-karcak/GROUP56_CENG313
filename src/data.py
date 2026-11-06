"""Data loading helpers for DRW Crypto Market Prediction (CENG313 Assignment 1)."""
import numpy as np
import pandas as pd

MARKET_COLS = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
TARGET = 'label'


def load_data(path, columns=None):
    """Read a parquet file and convert float64 columns to float32 to halve memory use."""
    df = pd.read_parquet(path, columns=columns)
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
    return df


def anonymous_cols(df):
    """Return the anonymous feature names (X1, X2, ...)."""
    return [c for c in df.columns if c.startswith('X')]


def clean_infinite(df):
    """Replace +inf / -inf with NaN so they are treated as missing values."""
    return df.replace([np.inf, -np.inf], np.nan)
