from pathlib import Path
import pandas as pd
from typing import Dict, Any, Tuple, List
from .config import *
from .paths import resolve_paths
from .data import (load_features, build_daily_from_index, ensure_complete_calendar,
                   rebuild_causal_rollings, add_calendar_rich, add_business_aggregates_tminus1,
                   build_causal_features, cap_outliers_if_needed)
from .lgbm_direct import filter_candidates, lgbm_importance_until, _fit_lgbm, _fit_lgbm_quantile
from .registry import model_dir, save_sklearn_like, save_pickle, save_keras, write_manifest
from .versioning import make_version_id

def data_pipeline(cwd: Path) -> Tuple[pd.DataFrame, Path, Path, Path, Path]:
    data_dir, results_dir, data_path, index_path = resolve_paths(cwd)
    df = load_features(data_path)
    if USE_INDEX_WEATHER_HOLIDAYS:
        daily_idx = build_daily_from_index(index_path, HOLIDAY_COL, WEATHER_AGG)
        if daily_idx is not None:
            df = df.merge(daily_idx, on="date", how="left")
    df = ensure_complete_calendar(df)
    df, _ = rebuild_causal_rollings(df, ycol=TARGET)
    df = cap_outliers_if_needed(df, ycol=TARGET, enabled=CAP_OUTLIERS, q=OUTLIER_Q)
    if USE_RICH_CALENDAR:
        df = add_calendar_rich(df, UA_HOLIDAYS_PATH)
    if ADD_BUSINESS_AGGREGATES:
        df = add_business_aggregates_tminus1(df, ycol=TARGET)
    df = build_causal_features(df, ycol=TARGET)
    return df, data_dir, results_dir, data_path, index_path

def train_final_lgbm_direct(df: pd.DataFrame, results_dir: Path, tag: str = None) -> Dict[str, Any]:
    # Selección de features sobre todo el historial (usando importancia contra y_1/y_7)
    feats = filter_candidates(df, target=TARGET, null_frac_max=0.30)
    df_l = df.sort_values(['product','date']).copy()
    for h in range(1, HORIZON+1):
        df_l[f'y_{h}'] = df_l.groupby('product')[TARGET].shift(-h)
    cutoff = df_l['date'].max()  # todo el historial
    imp_h1 = lgbm_importance_until(df_l, feats, "y_1", cutoff).head(TOPK_IMP)
    imp_h7 = lgbm_importance_until(df_l, feats, "y_7", cutoff).head(TOPK_IMP)
    sel = sorted(set(imp_h1['feature']).union(set(imp_h7['feature'])))
    # Entrenamos un modelo por horizonte (multi-modelo directo)
    models = {}
    objectives_to_try = ["auto"] if LGBM_OBJECTIVE == "auto" else [LGBM_OBJECTIVE]
    tweedie_grid = TWEEDIE_POWERS if "tweedie" in objectives_to_try else [None]

    for h in range(1, HORIZON+1):
        y_col = f"y_{h}"
        tr = df_l.dropna(subset=[y_col]).copy()
        X_tr, y_tr = tr[sel], tr[y_col]
        if USE_LOG1P_TARGET:
            import numpy as np
            y_tr = np.log1p(y_tr)
        best, best_mae = None, float('inf')
        import numpy as np
        for obj in objectives_to_try:
            for power in tweedie_grid:
                m = _fit_lgbm(X_tr, y_tr, objective=obj, tweedie_power=power)
                y_pred = m.predict(X_tr)
                if USE_LOG1P_TARGET:
                    y_pred = np.expm1(y_pred)
                cur = float(np.mean(np.abs(tr[y_col] - y_pred)))
                if cur < best_mae:
                    best_mae = cur; best = (m, obj, power)
        m_c, used_obj, used_power = best
        m_p10 = _fit_lgbm_quantile(X_tr, y_tr, alpha=0.10)
        m_p90 = _fit_lgbm_quantile(X_tr, y_tr, alpha=0.90)
        models[h] = {"center": m_c, "p10": m_p10, "p90": m_p90, "objective": used_obj, "tweedie_power": used_power}

    # Versionado + serialización
    version = make_version_id(
        params={
            "TARGET": TARGET, "HORIZON": HORIZON, "TOPK_IMP": TOPK_IMP,
            "USE_LOG1P_TARGET": USE_LOG1P_TARGET, "OBJECTIVE": LGBM_OBJECTIVE
        },
        tag=tag or "lgbm_direct"
    )
    out_dir = model_dir(results_dir, "lgbm_direct", version)
    # Guardamos un archivo por horizonte
    paths = []
    for h, bundle in models.items():
        h_dir = out_dir / f"h{h:02d}"
        h_dir.mkdir(parents=True, exist_ok=True)
        paths.append(save_sklearn_like(bundle["center"], h_dir, "center.joblib"))
        paths.append(save_sklearn_like(bundle["p10"], h_dir, "p10.joblib"))
        paths.append(save_sklearn_like(bundle["p90"], h_dir, "p90.joblib"))
    # Manifest
    write_manifest(out_dir, {
        "model_name": "lgbm_direct",
        "version": version,
        "features": sel,
        "objective_cfg": LGBM_OBJECTIVE,
        "use_log1p_target": USE_LOG1P_TARGET,
        "artifacts": [str(p) for p in paths]
    })
    return {"name": "lgbm_direct", "version": version, "path": out_dir, "features": sel}

def train_final_prophet(df: pd.DataFrame, results_dir: Path, tag: str = None) -> Dict[str, Any]:
    from prophet import Prophet
    regs = [c for c in ["is_holiday","wx_temperature_2m","wx_precipitation","wx_cloudcover",
                        "is_holiday_ext","is_holiday_prev","is_holiday_next"] if c in df.columns]
    version = make_version_id({"use_regressors": PROPHET_USE_REGRESSORS, "regs": regs}, tag or "prophet")
    out_dir = model_dir(results_dir, "prophet", version)
    # Entrenamos un modelo por producto (clásico en Prophet)
    artifacts = []
    for prod, g in df.sort_values("date").groupby("product"):
        aux = g.rename(columns={"date":"ds", TARGET:"y"}).copy()
        m = Prophet(interval_width=0.80, weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=False)
        if PROPHET_USE_REGRESSORS and regs:
            for r in regs: m.add_regressor(r)
        cols_fit = ["ds","y"] + (regs if PROPHET_USE_REGRESSORS else [])
        aux[cols_fit] = aux[cols_fit].sort_values("ds").ffill().bfill()
        m.fit(aux[cols_fit])
        pdir = out_dir / f"product={prod}"; pdir.mkdir(parents=True, exist_ok=True)
        artifacts.append(save_pickle(m, pdir, "model.pkl"))
    write_manifest(out_dir, {"model_name":"prophet","version":version,"regressors":regs,"artifacts":[str(a) for a in artifacts]})
    return {"name": "prophet", "version": version, "path": out_dir}

def train_final_sarimax(df: pd.DataFrame, results_dir: Path, tag: str = None) -> Dict[str, Any]:
    import statsmodels.api as sm
    exog_cols = [c for c in ["is_holiday","is_holiday_prev","is_holiday_next",
                             "wx_temperature_2m","wx_precipitation","wx_cloudcover"] if c in df.columns]
    version = make_version_id({"exog": exog_cols}, tag or "sarimax")
    out_dir = model_dir(results_dir, "sarimax", version)
    artifacts = []
    for prod, g in df.sort_values("date").groupby("product"):
        g = g.set_index("date")
        y_tr = g[TARGET].asfreq("D").ffill().bfill()
        ex_tr = g[exog_cols].asfreq("D").ffill().bfill().reindex(y_tr.index) if exog_cols else None
        orders_to_try = [((1,0,1),(1,0,1,7)), ((0,1,1),(0,1,1,7))]
        res = None
        for order, seas in orders_to_try:
            try:
                model = sm.tsa.statespace.SARIMAX(y_tr, order=order, seasonal_order=seas, exog=ex_tr,
                                                  enforce_stationarity=False, enforce_invertibility=False)
                res = model.fit(disp=False); break
            except Exception: 
                try:
                    model = sm.tsa.statespace.SARIMAX(y_tr, order=order, seasonal_order=seas, exog=None,
                                                      enforce_stationarity=False, enforce_invertibility=False)
                    res = model.fit(disp=False); break
                except Exception:
                    continue
        if res is None: 
            continue
        pdir = out_dir / f"product={prod}"; pdir.mkdir(parents=True, exist_ok=True)
        artifacts.append(save_pickle(res, pdir, "model.pkl"))
    write_manifest(out_dir, {"model_name":"sarimax","version":version,"exog_cols":exog_cols,"artifacts":[str(a) for a in artifacts]})
    return {"name": "sarimax", "version": version, "path": out_dir}

def train_final_lstm_direct(df: pd.DataFrame, results_dir: Path, tag: str = None) -> Dict[str, Any]:
    # Reutilizamos la arquitectura del módulo lstm_direct con entrenamiento único (no backtesting)
    from .lstm_direct import build_model as _build, numeric_feature_cols as _feats
    from sklearn.preprocessing import StandardScaler
    import numpy as np, keras

    feats = _feats(df, TARGET)
    df_ff = df.sort_values(['product','date']).copy()
    df_ff[feats] = df_ff[feats].ffill().bfill()

    # Construir dataset supervisado en TODO el historial para un único modelo global
    Xs, Ys = [], []
    for _, g in df_ff.groupby('product'):
        vals = g[feats].values
        tgt = g[TARGET].values.astype(float)
        for i in range(LSTM_LOOKBACK - 1, len(g) - HORIZON):
            Xs.append(vals[i - LSTM_LOOKBACK + 1 : i + 1, :])
            Ys.append(tgt[i + 1 : i + 1 + HORIZON])
    if not Xs:
        raise RuntimeError("No hay suficientes datos para LSTM.")
    X = np.stack(Xs); Y = np.stack(Ys)
    scaler = StandardScaler().fit(X.reshape(-1, X.shape[-1]))
    Xs = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype("float32")
    Yt = (np.log1p(Y) if USE_LOG1P_TARGET else Y).astype("float32")

    model = _build(n_feats=len(feats), horizon=HORIZON)
    es = keras.callbacks.EarlyStopping(monitor="loss", patience=LSTM_PATIENCE, restore_best_weights=True)
    model.fit(Xs, Yt, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH, verbose=0, callbacks=[es])

    version = make_version_id({"LOOKBACK":LSTM_LOOKBACK,"UNITS":LSTM_UNITS,"LR":LSTM_LR,"DROPOUT":LSTM_DROPOUT,
                               "USE_LOG1P_TARGET": USE_LOG1P_TARGET}, tag or "lstm_direct")
    out_dir = model_dir(results_dir, "lstm_direct", version)
    # Guardar modelo y scaler
    save_keras(model, out_dir, "model_keras")
    save_sklearn_like(scaler, out_dir, "scaler.joblib")
    write_manifest(out_dir, {"model_name":"lstm_direct","version":version,"features":feats})
    return {"name":"lstm_direct","version":version,"path":out_dir,"features":feats}
