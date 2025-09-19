import pandas as pd
import statsmodels.api as sm
from .config import TARGET, HORIZON, N_ORIGINS, MIN_TRAIN_DAYS
from .splits import rolling_origins
from .metrics import summarize_metrics

def run_sarimax(df):
    sarimax_all = []
    splits = rolling_origins(df["date"], n_origins=N_ORIGINS, horizon=HORIZON)
    exog_cols = [c for c in ["is_holiday","is_holiday_prev","is_holiday_next",
                             "wx_temperature_2m","wx_precipitation","wx_cloudcover"] if c in df.columns]
    print(f"[SARIMAX] exog_cols: {exog_cols}")

    for (train_end, test_start, test_end) in splits:
        train = df[df["date"] <= train_end].copy()
        test  = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
        n_days = train["date"].nunique()
        print(f"[SARIMAX] Split {train_end.date()} | train_days={n_days} | test_days={test['date'].nunique()}")
        if n_days < MIN_TRAIN_DAYS:
            print(f"[SARIMAX] Split saltado por historia insuficiente ({n_days} días).")
            continue

        fold_parts = []
        for prod, g in train.groupby("product"):
            g = g.sort_values("date").set_index("date")
            y_tr = g[TARGET].astype(float).asfreq("D").ffill().bfill()
            ex_tr = None
            if exog_cols:
                ex_tr = g[exog_cols].asfreq("D").ffill().bfill().reindex(y_tr.index)

            orders_to_try = [((1,0,1),(1,0,1,7)), ((0,1,1),(0,1,1,7))]
            res, chosen = None, None
            for order, seasonal_order in orders_to_try:
                try:
                    model = sm.tsa.statespace.SARIMAX(y_tr, order=order, seasonal_order=seasonal_order,
                                                      exog=ex_tr, enforce_stationarity=False, enforce_invertibility=False)
                    res = model.fit(disp=False); chosen = (order, seasonal_order, True); break
                except Exception:
                    try:
                        model = sm.tsa.statespace.SARIMAX(y_tr, order=order, seasonal_order=seasonal_order,
                                                          exog=None, enforce_stationarity=False, enforce_invertibility=False)
                        res = model.fit(disp=False); ex_tr = None; chosen = (order, seasonal_order, False); break
                    except Exception:
                        continue
            if res is None:
                print(f"[SARIMAX][{prod}] no se pudo ajustar en ningún orden.")
                continue

            future_idx = pd.date_range(test_start, test_end, freq="D")
            ex_te = None
            if exog_cols and chosen[2]:
                g_full = df[df["product"]==prod].set_index("date")
                ex_te = g_full[exog_cols].asfreq("D").ffill().bfill().reindex(future_idx)
            try:
                fcst = res.get_forecast(steps=len(future_idx), exog=ex_te)
                yhat = fcst.predicted_mean
                conf = fcst.conf_int(alpha=0.20)
                conf_cols = list(conf.columns)
                if len(conf_cols) >= 2:
                    lower_col = next((c for c in conf_cols if "lower" in c.lower()), conf_cols[0])
                    upper_col = next((c for c in conf_cols if "upper" in c.lower()), conf_cols[1])
                    yhat_p10 = conf[lower_col].values
                    yhat_p90 = conf[upper_col].values
                else:
                    yhat_p10 = None; yhat_p90 = None
            except Exception as e3:
                print(f"[SARIMAX][{prod}] fallo en forecast: {type(e3).__name__}: {e3}")
                continue

            out = pd.DataFrame({"date": future_idx, "product": prod, "yhat": yhat.values})
            if yhat_p10 is not None and yhat_p90 is not None:
                out["yhat_p10"] = yhat_p10; out["yhat_p90"] = yhat_p90
            fold_parts.append(out)

        if fold_parts:
            fold = pd.concat(fold_parts, ignore_index=True)
            fold["h"] = (fold["date"] - test_start).dt.days + 1
            y_true = test[["date","product",TARGET]].copy()
            merged = y_true.merge(fold, on=["date","product"], how="left")
            merged["model"] = "sarimax"
            sarimax_all.append(merged)
        else:
            print(f"[SARIMAX] Split {train_end.date()} sin predicciones válidas.")

    sarimax_results = (pd.concat(sarimax_all, ignore_index=True) if sarimax_all
                       else pd.DataFrame(columns=["date","product",TARGET,"h","yhat","yhat_p10","yhat_p90","model"]))
    if sarimax_results.empty:
        return sarimax_results, pd.DataFrame(), {}
    by_h_sarimax, overall_sarimax = summarize_metrics(sarimax_results.rename(columns={TARGET:"y"}))
    return sarimax_results, by_h_sarimax, overall_sarimax
