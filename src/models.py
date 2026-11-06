"""Validation and baseline models (CENG313 Assignment 1)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def pearson(y_true, y_pred):
    """Competition metric. Undefined (NaN) when the predictions are constant."""
    if np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def time_split(df, valid_frac=0.2):
    """Use the LAST part of the time series as validation (no shuffling, to avoid peeking at the future)."""
    df = df.sort_index()
    cut = int(len(df) * (1 - valid_frac))
    return df.iloc[:cut], df.iloc[cut:]


def evaluate(name, y_true, y_pred):
    """One row of the results table."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {'model': name, 'pearson': pearson(y_true, y_pred), 'rmse': rmse}


def fit_predict(model, train, valid, features, target):
    """Fill NaN with the TRAIN median, fit on train, predict on valid."""
    med = train[features].median()
    X_tr = train[features].fillna(med)
    X_va = valid[features].fillna(med)
    model.fit(X_tr, train[target])
    return model, model.predict(X_va), med


def linear_model():
    return LinearRegression()


def ridge_model(alpha=1.0):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
