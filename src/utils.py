"""EDA helper functions (CENG313 Assignment 1)."""
import numpy as np
import pandas as pd


def memory_mb(df):
    """Memory used by a DataFrame in MB."""
    return df.memory_usage(deep=True).sum() / 1024 ** 2


def missing_report(df):
    """Columns that have missing values: count and percentage, largest first."""
    counts = df.isnull().sum()
    report = pd.DataFrame({'missing': counts, 'percent': 100 * counts / len(df)})
    return report[report['missing'] > 0].sort_values('missing', ascending=False)


def infinite_report(df):
    """Number of +/-inf values per numeric column (only columns that have any)."""
    num = df.select_dtypes('number')
    counts = np.isinf(num).sum()
    return counts[counts > 0].sort_values(ascending=False)


def constant_columns(df):
    """Columns with a single unique value: they carry no information."""
    return [c for c in df.columns if df[c].nunique(dropna=False) <= 1]


def top_correlations(df, target, k=20):
    """Pearson correlation of every numeric feature with the target, top k by absolute value."""
    corr = df.drop(columns=[target]).corrwith(df[target])
    return corr.reindex(corr.abs().sort_values(ascending=False).index).head(k)
