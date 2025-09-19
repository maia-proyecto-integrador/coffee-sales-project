import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
import keras
from keras import layers

from .config import (
    TARGET, HORIZON, N_ORIGINS, MIN_TRAIN_DAYS,
    LSTM_LOOKBACK, LSTM_UNITS, LSTM_DROPOUT, LSTM_LR,
    LSTM_EPOCHS, LSTM_BATCH, LSTM_PATIENCE, USE_LOG1P_TARGET
)
from .splits import rolling_origins
from .metrics import summarize_metrics

from .registry import save_keras, save_sklearn_like, register_best_model


def numeric_feature_cols(dfin: pd.DataFrame, target: str) -> List[str]:
    cols = [c for c in dfin.columns if c not in {"date","product"} and not c.startswith("y_")
            and pd.api.types.is_numeric_dtype(dfin[c])]
    if target not in cols and target in dfin.columns:
        cols = [target] + cols
    return cols

def build_model(n_feats: int, horizon: int):
    inp = layers.Input(shape=(LSTM_LOOKBACK, n_feats))
    x = layers.LSTM(LSTM_UNITS, return_sequences=False)(inp)
    x = layers.Dropout(LSTM_DROPOUT)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(horizon)(x)
    m = keras.Model(inp, out)
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=LSTM_LR), loss="mse")
    return m

def _to_numpy(x):
    try:
        if hasattr(x, "detach"): return x.detach().cpu().numpy()
        return x.numpy()
    except Exception:
        return np.array(x)

def mc_dropout_predict_vec(model, X_win, n_samples=50):
    preds = []
    for _ in range(n_samples):
        y = model(X_win, training=True)
        y = _to_numpy(y).ravel()
        preds.append(y)
    P = np.stack(preds, axis=0)
    mean = P.mean(axis=0)
    p10  = np.percentile(P, 10, axis=0)
    p90  = np.percentile(P, 90, axis=0)
    return mean, p10, p90

def make_supervised_direct(df_train: pd.DataFrame, feats: List[str], target: str, lookback: int, horizon: int
                           ) -> Tuple[np.ndarray, np.ndarray]:
    Xs, Ys = [], []
    for _, g in df_train.sort_values(["product","date"]).groupby("product"):
        g = g.copy()
        g[feats] = g[feats].ffill().bfill()
        vals = g[feats].values
        tgt  = g[target].values.astype(float)
        for i in range(lookback - 1, len(g) - horizon):
            Xs.append(vals[i - lookback + 1 : i + 1, :])
            Ys.append(tgt[i + 1 : i + 1 + horizon])
    if not Xs:
        return np.empty((0, lookback, len(feats))), np.empty((0, horizon))
    return np.stack(Xs), np.stack(Ys)

def infer_direct_for_split(model, scaler, feats, train_prod, test_prod, target: str, lookback: int, horizon: int):
    gtr = train_prod.sort_values("date").copy()
    gte = test_prod.sort_values("date").copy()

    gtr[feats] = gtr[feats].ffill().bfill()
    gte[feats] = gte[feats].ffill().bfill()

    hist_vals = gtr[feats].values
    if len(hist_vals) == 0:
        hist_vals = gte[feats].values[:1, :]
    if len(hist_vals) < lookback:
        prepad = np.repeat(hist_vals[[0], :], lookback - len(hist_vals), axis=0)
        window = np.vstack([prepad, hist_vals])
    else:
        window = hist_vals[-lookback:, :]

    X_win = scaler.transform(window)
    X_win = X_win.reshape(1, lookback, len(feats)).astype("float32")

    yhat_vec, p10_vec, p90_vec = mc_dropout_predict_vec(model, X_win, n_samples=50)

    out = gte[["date"]].copy()
    out["h"] = (out["date"] - gte["date"].min()).dt.days + 1
    out["yhat"]     = [yhat_vec[h-1] if 1 <= h <= horizon else np.nan for h in out["h"]]
    out["yhat_p10"] = [p10_vec[h-1]  if 1 <= h <= horizon else np.nan for h in out["h"]]
    out["yhat_p90"] = [p90_vec[h-1]  if 1 <= h <= horizon else np.nan for h in out["h"]]
    out["product"]  = gte["product"].iloc[0]
    return out[["date","product","h","yhat","yhat_p10","yhat_p90"]]

def run_lstm_direct(
    df,
    persist_final: bool = True,
    register_as_dashboard_placeholder: bool = False  
):
    try:
        _ = keras.config.backend()
        TF_OK = True
    except Exception as e:
        print("[LSTM-direct] Backend Keras no disponible, se omite. Error:", e)
        TF_OK = False
    if not TF_OK:
        return (pd.DataFrame(columns=["date","product",TARGET,"h","yhat","yhat_p10","yhat_p90","model"]),
                pd.DataFrame(), {})

    base_feats = numeric_feature_cols(df, TARGET)
    if TARGET in df.columns and TARGET not in base_feats:
        base_feats = [TARGET] + base_feats
    base_feats = [c for c in base_feats if isinstance(c, str) and c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not base_feats:
        base_feats = numeric_feature_cols(df, TARGET)
    print(f"[LSTM] #features usadas: {len(base_feats)} → {base_feats[:15]}{' ...' if len(base_feats)>15 else ''}")

    lstm_dir_all = []
    last_model = None          
    last_scaler = None         
    splits = rolling_origins(df["date"], n_origins=N_ORIGINS, horizon=HORIZON)
    for (train_end, test_start, test_end) in splits:
        train = df[df["date"] <= train_end].copy()
        test  = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
        if train["date"].nunique() < MIN_TRAIN_DAYS:
            print(f"[LSTM-direct] Split saltado por historia insuficiente ({train['date'].nunique()} días).")
            continue

        df_train_ff = train.sort_values(["product","date"]).copy()
        df_train_ff[base_feats] = df_train_ff[base_feats].ffill().bfill()
        scaler = StandardScaler().fit(df_train_ff[base_feats].values)

        X_tr, Y_tr = make_supervised_direct(df_train_ff, base_feats, TARGET, LSTM_LOOKBACK, HORIZON)
        if X_tr.shape[0] == 0:
            print("[LSTM-direct] No se pudieron construir secuencias en este split.")
            continue

        X_tr_2d = X_tr.reshape(-1, X_tr.shape[-1])
        X_tr_scaled = scaler.transform(X_tr_2d).reshape(X_tr.shape).astype("float32")
        Y_tr_model  = (np.log1p(Y_tr) if USE_LOG1P_TARGET else Y_tr).astype("float32")

        model = build_model(n_feats=len(base_feats), horizon=HORIZON)
        es = keras.callbacks.EarlyStopping(monitor="loss", patience=LSTM_PATIENCE, restore_best_weights=True)
        model.fit(X_tr_scaled, Y_tr_model, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH, verbose=0, callbacks=[es])

        
        last_model = model
        last_scaler = scaler

        fold_parts = []
        for prod, gte in test.groupby("product"):
            gtr = train[train["product"] == prod].copy()
            pred_df = infer_direct_for_split(model, scaler, base_feats, gtr, gte, TARGET, LSTM_LOOKBACK, HORIZON)
            if USE_LOG1P_TARGET:
                for c in ["yhat","yhat_p10","yhat_p90"]:
                    if c in pred_df.columns:
                        pred_df[c] = np.expm1(pred_df[c])
            merged = gte[["date","product",TARGET]].merge(pred_df, on=["date","product"], how="left")
            fold_parts.append(merged)
        merged_fold = (pd.concat(fold_parts, ignore_index=True) if fold_parts
                       else pd.DataFrame(columns=["date","product",TARGET,"h","yhat","yhat_p10","yhat_p90"]))
        merged_fold["model"] = "lstm_direct"
        lstm_dir_all.append(merged_fold[["date","product",TARGET,"h","yhat","yhat_p10","yhat_p90","model"]])

    lstm_direct_results = (pd.concat(lstm_dir_all, ignore_index=True)
                           if lstm_dir_all else pd.DataFrame(columns=["date","product",TARGET,"h","yhat","yhat_p10","yhat_p90","model"]))
    by_h_lstm_dir, overall_lstm_dir = summarize_metrics(lstm_direct_results.rename(columns={TARGET:"y"}))

    
    if persist_final and last_model is not None and last_scaler is not None:
        try:
            keras_path = save_keras(last_model, out_dir=None if False else None)  # usa default de registry (models/)
        except TypeError:
            # Si tu save_keras requiere out_dir obligatorio, usa:
            from .registry import MODELS_DIR
            keras_path = save_keras(last_model, out_dir=MODELS_DIR, filename="lstm_direct")
        _ = save_sklearn_like(last_scaler, out_dir=keras_path if keras_path.is_dir() else keras_path.parent, filename="scaler.joblib")
        print(f"[LSTM-direct] Artefactos guardados en: {keras_path}")

    
    if register_as_dashboard_placeholder:
        metric_name = "sMAPE"
        best_score = float(overall_lstm_dir.get(metric_name, 0.0)) if isinstance(overall_lstm_dir, dict) else 0.0
        bundle = {
            "family": "lstm_direct",
            "note": "placeholder para dashboard (usa promedio 7d). El artefacto Keras y scaler se guardaron aparte.",
        }
        entry = register_best_model(
            model_obj=bundle,
            name="lgbm_direct",   # << para que el dashboard use su fallback y NO se rompa
            metric=metric_name,
            score=best_score,
            horizon=HORIZON
        )
        print("[LSTM-direct] best_model (placeholder) actualizado:", entry)

    return lstm_direct_results, by_h_lstm_dir, overall_lstm_dir

