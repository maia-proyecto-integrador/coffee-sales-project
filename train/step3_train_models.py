# ===========================
# 📦 Importaciones
# ===========================
import os
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
from lightgbm import LGBMRegressor
from prophet import Prophet
import statsmodels.api as sm
import keras
from keras import layers
from sklearn.preprocessing import StandardScaler

from step2_setup_mlflow import mae, rmse, mape, smape
from mlflow_utils import _mlflow_log_run

np.random.seed(42)

# ===========================
# 📂 Paths
# ===========================
RESULTS_DIR = Path("results")
ARTIFACTS_DIR = RESULTS_DIR / "artifacts"
DATA_PATH = ARTIFACTS_DIR / "df_final.csv"

TARGET = "transactions"
HORIZON = 7
N_ORIGINS = 4
MIN_TRAIN_DAYS = 150
TOPK_IMP = 40
USE_LOG1P_TARGET = False
RANDOM_STATE = 42

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
print(f"✅ Dataset cargado: {df.shape}")

# ===========================
# ⚙️ Utilidades
# ===========================
def rolling_origins(date_index: pd.Series, n_origins: int = 4, horizon: int = 7):
    unique_dates = pd.Series(pd.to_datetime(pd.unique(date_index))).sort_values()
    anchors = [unique_dates.iloc[-(i+1)*horizon] for i in range(n_origins)][::-1]
    splits = []
    for anchor in anchors:
        train_end = anchor - pd.Timedelta(days=1)
        test_start = anchor
        test_end = anchor + pd.Timedelta(days=horizon - 1)
        splits.append((train_end, test_start, test_end))
    return splits

def summarize_metrics(df_eval, target_col="y"):
    overall = {
        "MAE": mae(df_eval[target_col], df_eval["yhat"]),
        "RMSE": rmse(df_eval[target_col], df_eval["yhat"]),
        "MAPE": mape(df_eval[target_col], df_eval["yhat"]),
        "sMAPE": smape(df_eval[target_col], df_eval["yhat"]),
    }
    return pd.DataFrame([overall])

# ===========================
# 1️⃣ Baselines
# ===========================
BASELINE_NAMES = ["naive1", "snaive7", "ma7"]

def _baseline_predict(train, test_dates, horizon, target, kind):
    pieces = []
    for prod, g in train.groupby("product"):
        g = g.sort_values("date")
        if kind == "naive1":
            preds = [g[target].iloc[-1]] * horizon
        elif kind == "snaive7":
            hist = g[target].iloc[-7:].tolist()
            preds = hist if len(hist) == 7 else [g[target].iloc[-1]] * horizon
        elif kind == "ma7":
            preds = [g[target].iloc[-7:].mean()] * horizon
        dfp = pd.DataFrame({
            "date": test_dates,
            "product": prod,
            "h": list(range(1, horizon+1)),
            "yhat": preds[:horizon]
        })
        pieces.append(dfp)
    return pd.concat(pieces, ignore_index=True)

baseline_all = []
splits = rolling_origins(df["date"], n_origins=N_ORIGINS, horizon=HORIZON)
for (train_end, test_start, test_end) in splits:
    train = df[df["date"] <= train_end].copy()
    test = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
    horizon_idx = pd.date_range(test_start, test_end, freq="D")
    for name in BASELINE_NAMES:
        preds = _baseline_predict(train, horizon_idx, HORIZON, TARGET, name)
        merged = test.merge(preds, on=["date","product"], how="left")
        merged["model"] = name
        baseline_all.append(merged[["date","product",TARGET,"h","yhat","model"]])
baseline_results = pd.concat(baseline_all, ignore_index=True)
overall_baseline = summarize_metrics(baseline_results.rename(columns={TARGET:"y"}))
baseline_results.to_csv(RESULTS_DIR / "baselines_forecasts.csv", index=False)
overall_baseline.to_csv(RESULTS_DIR / "baselines_metrics_overall.csv", index=False)
_mlflow_log_run("baselines", {"HORIZON": HORIZON},
                RESULTS_DIR/"baselines_metrics_overall.csv",
                RESULTS_DIR/"baselines_metrics_overall.csv",
                [RESULTS_DIR/"baselines_forecasts.csv"],
                {"family":"baselines"})

# ===========================
# 2️⃣ LightGBM directo
# ===========================
candidates = [c for c in df.columns if c not in {"date","product",TARGET}]

df_l = df.copy()
for h in range(1, HORIZON+1):
    df_l[f"y_{h}"] = df_l.groupby("product")[TARGET].shift(-h)

lgbm_all = []
splits = rolling_origins(df_l["date"], n_origins=N_ORIGINS, horizon=HORIZON)
for (train_end, test_start, test_end) in splits:
    train = df_l[df_l["date"] <= train_end].copy()
    test  = df_l[(df_l["date"] >= test_start) & (df_l["date"] <= test_end)].copy()
    preds_blocks = []
    for h in range(1, HORIZON+1):
        y_col = f"y_{h}"
        tr = train.dropna(subset=[y_col])
        if tr.empty:
            continue
        X_tr, y_tr = tr[candidates], tr[y_col]
        m = LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=RANDOM_STATE)
        m.fit(X_tr, y_tr)
        test_block = test.copy()
        test_block["h"] = (test_block["date"] - test_start).dt.days + 1
        mask_h = test_block["h"] == h
        X_te = test_block.loc[mask_h, candidates]
        yhat = m.predict(X_te)
        out = test_block.loc[mask_h, ["date","product"]].copy()
        out["h"] = h
        out["yhat"] = yhat
        preds_blocks.append(out)
    if preds_blocks:
        fold_preds = pd.concat(preds_blocks, ignore_index=True)
        y_true = test[["date","product",TARGET]]
        merged = y_true.merge(fold_preds, on=["date","product"], how="left")
        merged["model"] = "lgbm_direct"
        lgbm_all.append(merged)
lgbm_results = pd.concat(lgbm_all, ignore_index=True)
overall_lgbm = summarize_metrics(lgbm_results.rename(columns={TARGET:"y"}))
lgbm_results.to_csv(RESULTS_DIR / "lgbm_direct_forecasts.csv", index=False)
overall_lgbm.to_csv(RESULTS_DIR / "lgbm_direct_metrics_overall.csv", index=False)
_mlflow_log_run("lgbm_direct", {"HORIZON": HORIZON},
                RESULTS_DIR/"lgbm_direct_metrics_overall.csv",
                RESULTS_DIR/"lgbm_direct_metrics_overall.csv",
                [RESULTS_DIR/"lgbm_direct_forecasts.csv"],
                {"family":"lgbm"})

# ===========================
# 3️⃣ Prophet
# ===========================
prophet_all = []
splits = rolling_origins(df["date"], n_origins=N_ORIGINS, horizon=HORIZON)
for (train_end, test_start, test_end) in splits:
    train = df[df["date"] <= train_end].copy()
    test  = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
    fold = []
    for prod, g in train.groupby("product"):
        aux = g.rename(columns={"date":"ds", TARGET:"y"})
        m = Prophet(weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=False)
        m.fit(aux[["ds","y"]])
        future = pd.DataFrame({"ds": pd.date_range(test_start, test_end, freq="D")})
        fcst = m.predict(future)[["ds","yhat"]]
        fcst["product"] = prod
        fold.append(fcst.rename(columns={"ds":"date"}))
    fold = pd.concat(fold, ignore_index=True)
    fold["h"] = (fold["date"] - test_start).dt.days + 1
    y_true = test[["date","product",TARGET]]
    merged = y_true.merge(fold, on=["date","product"], how="left")
    merged["model"] = "prophet"
    prophet_all.append(merged)
prophet_results = pd.concat(prophet_all, ignore_index=True)
overall_prophet = summarize_metrics(prophet_results.rename(columns={TARGET:"y"}))
prophet_results.to_csv(RESULTS_DIR / "prophet_forecasts.csv", index=False)
overall_prophet.to_csv(RESULTS_DIR / "prophet_metrics_overall.csv", index=False)
_mlflow_log_run("prophet", {"HORIZON": HORIZON},
                RESULTS_DIR/"prophet_metrics_overall.csv",
                RESULTS_DIR/"prophet_metrics_overall.csv",
                [RESULTS_DIR/"prophet_forecasts.csv"],
                {"family":"prophet"})

# ===========================
# 4️⃣ SARIMAX
# ===========================
sarimax_all = []
splits = rolling_origins(df["date"], n_origins=N_ORIGINS, horizon=HORIZON)
for (train_end, test_start, test_end) in splits:
    train = df[df["date"] <= train_end].copy()
    test  = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
    fold_parts = []
    for prod, g in train.groupby("product"):
        y_tr = g.set_index("date")[TARGET].asfreq("D").fillna(0)
        try:
            model = sm.tsa.statespace.SARIMAX(y_tr, order=(1,1,1), seasonal_order=(1,1,1,7))
            res = model.fit(disp=False)
            future_idx = pd.date_range(test_start, test_end, freq="D")
            yhat = res.forecast(steps=len(future_idx))
            out = pd.DataFrame({"date": future_idx, "product": prod, "yhat": yhat.values})
            fold_parts.append(out)
        except Exception:
            continue
    if fold_parts:
        fold = pd.concat(fold_parts, ignore_index=True)
        fold["h"] = (fold["date"] - test_start).dt.days + 1
        y_true = test[["date","product",TARGET]]
        merged = y_true.merge(fold, on=["date","product"], how="left")
        merged["model"] = "sarimax"
        sarimax_all.append(merged)
sarimax_results = pd.concat(sarimax_all, ignore_index=True)
overall_sarimax = summarize_metrics(sarimax_results.rename(columns={TARGET:"y"}))
sarimax_results.to_csv(RESULTS_DIR / "sarimax_forecasts.csv", index=False)
overall_sarimax.to_csv(RESULTS_DIR / "sarimax_metrics_overall.csv", index=False)
_mlflow_log_run("sarimax", {"HORIZON": HORIZON},
                RESULTS_DIR/"sarimax_metrics_overall.csv",
                RESULTS_DIR/"sarimax_metrics_overall.csv",
                [RESULTS_DIR/"sarimax_forecasts.csv"],
                {"family":"sarimax"})

# ===========================
# 5️⃣ LSTM directo
# ===========================
LSTM_LOOKBACK = 30
LSTM_UNITS = 64
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 10
LSTM_BATCH = 128
LSTM_LR = 1e-3

def _numeric_feats(dfin):
    return [c for c in dfin.columns if c not in {"date","product"} and pd.api.types.is_numeric_dtype(dfin[c])]

def _make_supervised(df_train, feats, target, lookback, horizon):
    Xs, Ys = [], []
    for _, g in df_train.groupby("product"):
        vals = g[feats].values
        tgt  = g[target].values
        for i in range(lookback-1, len(g)-horizon):
            Xs.append(vals[i-lookback+1:i+1])
            Ys.append(tgt[i+1:i+1+horizon])
    return np.stack(Xs), np.stack(Ys)

def _build_lstm_model(n_feats, horizon):
    inp = layers.Input(shape=(LSTM_LOOKBACK, n_feats))
    x = layers.LSTM(LSTM_UNITS)(inp)
    x = layers.Dropout(LSTM_DROPOUT)(x)
    out = layers.Dense(horizon)(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=LSTM_LR), loss="mse")
    return m

lstm_all = []
splits = rolling_origins(df["date"], n_origins=N_ORIGINS, horizon=HORIZON)
for (train_end, test_start, test_end) in splits:
    train = df[df["date"] <= train_end].copy()
    test  = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
    feats = _numeric_feats(df)
    X_tr, Y_tr = _make_supervised(train, feats, TARGET, LSTM_LOOKBACK, HORIZON)
    if X_tr.size == 0:
        continue
    scaler = StandardScaler().fit(X_tr.reshape(-1, X_tr.shape[-1]))
    X_tr_scaled = scaler.transform(X_tr.reshape(-1, X_tr.shape[-1])).reshape(X_tr.shape)
    model = _build_lstm_model(len(feats), HORIZON)
    model.fit(X_tr_scaled, Y_tr, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH, verbose=0)
    fold_parts = []
    for prod, g in test.groupby("product"):
        g_train = train[train["product"] == prod]
        vals = g_train[feats].tail(LSTM_LOOKBACK).values
        if len(vals) < LSTM_LOOKBACK:
            continue
        X_win = scaler.transform(vals).reshape(1,LSTM_LOOKBACK,len(feats))
        yhat_vec = model.predict(X_win)[0]
        out = pd.DataFrame({
            "date": pd.date_range(test_start, periods=HORIZON, freq="D"),
            "product": prod,
            "h": range(1,HORIZON+1),
            "yhat": yhat_vec
        })
        fold_parts.append(out)
    if fold_parts:
        fold = pd.concat(fold_parts, ignore_index=True)
        y_true = test[["date","product",TARGET]]
        merged = y_true.merge(fold, on=["date","product"], how="left")
        merged["model"] = "lstm_direct"
        lstm_all.append(merged)
lstm_results = pd.concat(lstm_all, ignore_index=True)
overall_lstm = summarize_metrics(lstm_results.rename(columns={TARGET:"y"}))
lstm_results.to_csv(RESULTS_DIR / "lstm_direct_forecasts.csv", index=False)
overall_lstm.to_csv(RESULTS_DIR / "lstm_direct_metrics_overall.csv", index=False)
_mlflow_log_run("lstm_direct", {"HORIZON": HORIZON},
                RESULTS_DIR/"lstm_direct_metrics_overall.csv",
                RESULTS_DIR/"lstm_direct_metrics_overall.csv",
                [RESULTS_DIR/"lstm_direct_forecasts.csv"],
                {"family":"lstm"})

# ===========================
# 6️⃣ Ensemble
# ===========================
all_models = {
    "baselines": pd.read_csv(RESULTS_DIR / "baselines_forecasts.csv"),
    "lgbm_direct": pd.read_csv(RESULTS_DIR / "lgbm_direct_forecasts.csv"),
    "prophet": pd.read_csv(RESULTS_DIR / "prophet_forecasts.csv"),
    "sarimax": pd.read_csv(RESULTS_DIR / "sarimax_forecasts.csv"),
    "lstm_direct": pd.read_csv(RESULTS_DIR / "lstm_direct_forecasts.csv"),
}
all_overall = []
for name, dfp in all_models.items():
    dfp2 = dfp.rename(columns={TARGET:"y"})
    overall = summarize_metrics(dfp2)
    overall["model"] = name
    all_overall.append(overall)
rank = pd.concat(all_overall, ignore_index=True).sort_values("sMAPE")
rank.to_csv(RESULTS_DIR / "_model_ranking.csv", index=False)
_mlflow_log_run("ensemble", {"strategy": "ranking-smape"},
                RESULTS_DIR/"_model_ranking.csv",
                RESULTS_DIR/"_model_ranking.csv",
                [RESULTS_DIR/"_model_ranking.csv"],
                {"family":"ensemble"})

print("✅ TODOS los modelos entrenados y registrados en MLflow")
