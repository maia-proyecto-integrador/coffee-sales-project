import pandas as pd
from .config import TARGET, HORIZON, MIN_TRAIN_DAYS
from .splits import rolling_origins
from .metrics import summarize_metrics

BASELINE_NAMES = ["naive1", "snaive7", "ma7"]

def _baseline_predict(train: pd.DataFrame, test_dates: pd.DatetimeIndex, horizon: int, target: str, kind: str):
    pieces = []
    for prod, g in train.groupby("product"):
        g = g.sort_values("date")
        if len(g) == 0: continue
        if kind == "naive1":
            last = g[target].iloc[-1]; preds = [last] * horizon
        elif kind == "snaive7":
            hist = g[target].iloc[-7:].tolist()
            if len(hist) < 7:
                hist = [g[target].iloc[-1]] * 7
            preds = hist
        elif kind == "ma7":
            window = g[target].iloc[-7:]
            meanv = float(window.mean()) if len(window) > 0 else float(g[target].iloc[-1])
            preds = [meanv] * horizon
        else:
            raise ValueError(kind)
        dfp = pd.DataFrame({
            "date": test_dates,
            "product": prod,
            "h": list(range(1, horizon + 1)),
            "yhat": preds[:horizon],
        })
        pieces.append(dfp)
    return pd.concat(pieces, ignore_index=True)

def run_baselines(df, results_dir):
    baseline_all = []
    splits = rolling_origins(df["date"], n_origins=4, horizon=HORIZON)
    for (train_end, test_start, test_end) in splits:
        train = df[df["date"] <= train_end].copy()
        test  = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
        test["date"] = pd.to_datetime(test["date"])

        if train["date"].nunique() < MIN_TRAIN_DAYS:
            print(f"[Baselines] Split saltado por historia insuficiente ({train['date'].nunique()} días).")
            continue

        horizon_idx = pd.date_range(test_start, test_end, freq="D")
        for name in BASELINE_NAMES:
            preds = _baseline_predict(train, horizon_idx, HORIZON, TARGET, name).copy()
            if "date" not in preds:
                raise ValueError("[Baselines] predictor debe devolver 'date'.")
            if "product" not in preds.columns:
                raise ValueError("[Baselines] predictor debe devolver 'product'.")
            if "yhat" not in preds.columns:
                raise ValueError("[Baselines] predictor debe devolver 'yhat'.")
            if "h" not in preds.columns:
                preds["h"] = (preds["date"] - test_start).dt.days + 1
            merged = test.merge(preds[["date","product","h","yhat"]], on=["date","product"], how="left")
            merged["model"] = name
            baseline_all.append(merged[["date","product",TARGET,"h","yhat","model"]])

    baseline_results = (pd.concat(baseline_all, ignore_index=True)
                        if baseline_all else
                        pd.DataFrame(columns=["date","product",TARGET,"h","yhat","model"]))
    by_h_baseline, overall_baseline = summarize_metrics(baseline_results.rename(columns={TARGET:"y"}))
    return baseline_results, by_h_baseline, overall_baseline
