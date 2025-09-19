import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from .config import (TARGET, HORIZON, N_ORIGINS, MIN_TRAIN_DAYS, TOPK_IMP,
                     USE_LOG1P_TARGET, LGBM_OBJECTIVE, TWEEDIE_POWERS, RANDOM_STATE)
from .splits import rolling_origins
from .metrics import summarize_metrics
from .registry import register_best_model   # ⬅️ NUEVO

def filter_candidates(dfin, target=TARGET, null_frac_max=0.30):
    drop_like = {target, "date", "product"}
    cols = [c for c in dfin.columns if c not in drop_like and not c.startswith("y_")]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(dfin[c])]
    kept = []
    for c in cols:
        null_frac = dfin[c].isna().mean()
        if null_frac > null_frac_max: 
            continue
        if dfin[c].nunique(dropna=True) <= 1:
            continue
        kept.append(c)
    return kept

def build_direct_labels(dfin, target, H=7):
    dfx = dfin.sort_values(["product","date"]).copy()
    for h in range(1, H+1):
        dfx[f"y_{h}"] = dfx.groupby("product")[target].shift(-h)
    return dfx

def _fit_lgbm(X, y, objective="auto", tweedie_power=None):
    params = dict(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE
    )
    if objective == "poisson":
        params.update(objective="poisson")
    elif objective == "tweedie":
        params.update(objective="tweedie")
        if tweedie_power is not None:
            params.update(tweedie_variance_power=float(tweedie_power))
    model = LGBMRegressor(**params)
    model.fit(X, y)
    return model

def _fit_lgbm_quantile(X, y, alpha):
    model = LGBMRegressor(
        objective="quantile", alpha=alpha,
        n_estimators=500, learning_rate=0.05,
        num_leaves=31, subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE
    )
    model.fit(X, y)
    return model

def lgbm_importance_until(dfin, feats, y_col, cutoff_date):
    data = dfin[(dfin["date"] <= cutoff_date)].dropna(subset=[y_col]).copy()
    if data.empty:
        return pd.DataFrame({"feature":[], "importance":[]})
    X, y = data[feats], data[y_col]
    m = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                      subsample=0.9, colsample_bytree=0.9, random_state=RANDOM_STATE)
    m.fit(X, y)
    imp = pd.DataFrame({"feature": feats, "importance": m.feature_importances_})
    return imp.sort_values("importance", ascending=False)

def run_lgbm(df, register_final: bool = True):
    candidates = filter_candidates(df, target=TARGET, null_frac_max=0.30)
    df_l = build_direct_labels(df, TARGET, H=HORIZON)

    lgbm_all = []
    last_selected_feats = None  # ⬅️ NUEVO: guardamos las últimas features elegidas
    splits = rolling_origins(df_l["date"], n_origins=N_ORIGINS, horizon=HORIZON)
    for (train_end, test_start, test_end) in splits:
        train = df_l[df_l["date"] <= train_end].copy()
        test  = df_l[(df_l["date"] >= test_start) & (df_l["date"] <= test_end)].copy()

        if train["date"].nunique() < MIN_TRAIN_DAYS:
            print(f"[LGBM] Split saltado: historia insuficiente ({train['date'].nunique()} días).")
            continue

        imp_h1 = lgbm_importance_until(train, candidates, "y_1", train_end).head(TOPK_IMP)
        imp_h7 = lgbm_importance_until(train, candidates, "y_7", train_end).head(TOPK_IMP)
        selected_feats = sorted(set(imp_h1["feature"]).union(set(imp_h7["feature"])))
        last_selected_feats = selected_feats if selected_feats else last_selected_feats  # ⬅️ NUEVO

        preds_blocks = []
        for h in range(1, HORIZON + 1):
            y_col = f"y_{h}"
            tr = train.dropna(subset=[y_col]).copy()
            if tr.empty: continue

            X_tr, y_tr = tr[selected_feats], tr[y_col]
            if USE_LOG1P_TARGET:
                y_tr = np.log1p(y_tr)

            objectives_to_try = ["auto"]
            if LGBM_OBJECTIVE in ["poisson", "tweedie"]:
                objectives_to_try = [LGBM_OBJECTIVE]
            tweedie_grid = TWEEDIE_POWERS if "tweedie" in objectives_to_try else [None]

            best_model, best_mae = None, np.inf
            for obj in objectives_to_try:
                for power in tweedie_grid:
                    mtmp = _fit_lgbm(X_tr, y_tr, objective=obj, tweedie_power=power)
                    y_pred_tr = mtmp.predict(X_tr)
                    if USE_LOG1P_TARGET:
                        y_pred_tr = np.expm1(y_pred_tr)
                    cur_mae = float(np.mean(np.abs(tr[y_col] - y_pred_tr)))
                    if cur_mae < best_mae:
                        best_mae = cur_mae
                        best_model = (mtmp, obj, power)

            model_c, used_obj, used_power = best_model
            model_p10 = _fit_lgbm_quantile(X_tr, y_tr, alpha=0.10)
            model_p90 = _fit_lgbm_quantile(X_tr, y_tr, alpha=0.90)

            test_block = test.copy()
            test_block["h"] = (test_block["date"] - test_start).dt.days + 1
            mask_h = test_block["h"] == h
            X_te = test_block.loc[mask_h, selected_feats]

            yhat_c   = model_c.predict(X_te)
            yhat_p10 = model_p10.predict(X_te)
            yhat_p90 = model_p90.predict(X_te)

            if USE_LOG1P_TARGET:
                yhat_c   = np.expm1(yhat_c)
                yhat_p10 = np.expm1(yhat_p10)
                yhat_p90 = np.expm1(yhat_p90)

            out = test_block.loc[mask_h, ["date", "product"]].copy()
            out["h"] = h
            out["yhat"] = yhat_c
            out["yhat_p10"] = yhat_p10
            out["yhat_p90"] = yhat_p90
            out["lgbm_objective"] = used_obj
            out["tweedie_power"] = used_power
            preds_blocks.append(out)

        if preds_blocks:
            fold_preds = pd.concat(preds_blocks, ignore_index=True)
            y_true = test[["date", "product", TARGET]].copy()
            merged = y_true.merge(fold_preds, on=["date", "product"], how="left")
            merged["model"] = "lgbm_direct"
            lgbm_all.append(merged)

    lgbm_results = (pd.concat(lgbm_all, ignore_index=True) if lgbm_all
                    else pd.DataFrame(columns=["date","product",TARGET,"h","yhat","yhat_p10","yhat_p90","model"]))
    by_h_lgbm, overall_lgbm = summarize_metrics(lgbm_results.rename(columns={TARGET: "y"}))

    # ---------------------------
    # REGISTRO COMPATIBLE (opcional)
    # ---------------------------
    if register_final:
        # 1) Elegimos features finales
        final_feats = last_selected_feats if last_selected_feats else candidates
        if not final_feats:
            final_feats = candidates  # fallback

        # 2) Refit final por horizonte sobre toda la historia disponible
        df_full = df_l.copy()
        bundle_models = {}
        objectives_to_try = ["auto"] if LGBM_OBJECTIVE not in ["poisson", "tweedie"] else [LGBM_OBJECTIVE]
        tweedie_grid = TWEEDIE_POWERS if "tweedie" in objectives_to_try else [None]

        for h in range(1, HORIZON + 1):
            y_col = f"y_{h}"
            full_tr = df_full.dropna(subset=[y_col]).copy()
            if full_tr.empty:
                continue

            X_tr, y_tr = full_tr[final_feats], full_tr[y_col]
            if USE_LOG1P_TARGET:
                y_tr = np.log1p(y_tr)

            best_model, best_mae = None, np.inf
            best_obj, best_power = "auto", None
            for obj in objectives_to_try:
                for power in tweedie_grid:
                    mtmp = _fit_lgbm(X_tr, y_tr, objective=obj, tweedie_power=power)
                    y_pred_tr = mtmp.predict(X_tr)
                    y_pred_inv = np.expm1(y_pred_tr) if USE_LOG1P_TARGET else y_pred_tr
                    cur_mae = float(np.mean(np.abs(full_tr[y_col] - y_pred_inv)))
                    if cur_mae < best_mae:
                        best_mae = cur_mae
                        best_model = mtmp
                        best_obj, best_power = obj, power

            model_c = best_model
            model_p10 = _fit_lgbm_quantile(X_tr, y_tr, alpha=0.10)
            model_p90 = _fit_lgbm_quantile(X_tr, y_tr, alpha=0.90)

            bundle_models[h] = {
                "center": model_c,
                "p10": model_p10,
                "p90": model_p90,
                "objective": best_obj,
                "tweedie_power": best_power
            }

        # 3) Armamos bundle y registramos (el dashboard lo aceptará y usará su placeholder)
        bundle = {
            "family": "lgbm_direct",
            "horizon": HORIZON,
            "features": list(final_feats),
            "models": bundle_models,
            "use_log1p_target": bool(USE_LOG1P_TARGET)
        }

        metric_name = "sMAPE"
        best_score = float(overall_lgbm.get(metric_name, 0.0)) if isinstance(overall_lgbm, dict) else 0.0

        entry = register_best_model(
            model_obj=bundle,
            name="lgbm_direct",   # ← el dashboard ve 'lgbm' y aplica su fallback actual
            metric=metric_name,
            score=best_score,
            horizon=HORIZON
        )
        print("[LGBM] best_model registrado (bundle):", entry)

    return lgbm_results, by_h_lgbm, overall_lgbm, candidates
