import numpy as np
import pandas as pd

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0: return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0: return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2)))

def mape(y_true, y_pred, epsilon=1e-6, ignore_zeros=True):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if ignore_zeros:
        mask = np.abs(y_true) > epsilon
        if mask.sum() == 0: return np.nan
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))*100.0)
    else:
        denom = np.where(np.abs(y_true) < epsilon, epsilon, np.abs(y_true))
        return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

def smape(y_true, y_pred, epsilon=1e-6):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)).clip(min=epsilon)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)

def coverage(y_true, y_lo, y_hi):
    y_true = np.asarray(y_true, dtype=float)
    y_lo = np.asarray(y_lo, dtype=float)
    y_hi = np.asarray(y_hi, dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_lo) & ~np.isnan(y_hi)
    if mask.sum() == 0: return np.nan
    inside = (y_true[mask] >= y_lo[mask]) & (y_true[mask] <= y_hi[mask])
    return float(np.mean(inside) * 100.0)

def summarize_metrics(df_eval, target_col="y"):
    out = {}
    for h, g in df_eval.groupby("h"):
        out[h] = {
            "MAE": mae(g[target_col], g["yhat"]),
            "RMSE": rmse(g[target_col], g["yhat"]),
            "MAPE": mape(g[target_col], g["yhat"], ignore_zeros=True),
            "sMAPE": smape(g[target_col], g["yhat"]),
        }
        if {"yhat_p10","yhat_p90"}.issubset(g.columns):
            out[h]["COV_p10_p90_%"] = coverage(g[target_col], g["yhat_p10"], g["yhat_p90"])
    overall = {
        "MAE": mae(df_eval[target_col], df_eval["yhat"]),
        "RMSE": rmse(df_eval[target_col], df_eval["yhat"]),
        "MAPE": mape(df_eval[target_col], df_eval["yhat"], ignore_zeros=True),
        "sMAPE": smape(df_eval[target_col], df_eval["yhat"]),
    }
    if {"yhat_p10","yhat_p90"}.issubset(df_eval.columns):
        overall["COV_p10_p90_%"] = coverage(df_eval[target_col], df_eval["yhat_p10"], df_eval["yhat_p90"])
    return pd.DataFrame(out).T, overall
