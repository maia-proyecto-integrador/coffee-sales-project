import pandas as pd
from prophet import Prophet
from .config import (TARGET, HORIZON, N_ORIGINS, MIN_TRAIN_DAYS, PROPHET_USE_REGRESSORS)
from .splits import rolling_origins
from .metrics import summarize_metrics
from .registry import register_best_model   #
import numpy as np                          

def run_prophet(df, register_final: bool = True):  
    candidate_regressors = [c for c in ["is_holiday","wx_temperature_2m","wx_precipitation","wx_cloudcover",
                                        "is_holiday_ext","is_holiday_prev","is_holiday_next"] if c in df.columns]
    print("[Prophet] Regresores detectados:", candidate_regressors)

    prophet_all = []
    splits = rolling_origins(df["date"], n_origins=N_ORIGINS, horizon=HORIZON)
    for (train_end, test_start, test_end) in splits:
        train = df[df["date"] <= train_end].copy()
        test  = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
        if train["date"].nunique() < MIN_TRAIN_DAYS:
            print(f"[Prophet] Split saltado por historia insuficiente ({train['date'].nunique()} días).")
            continue

        fold = []
        for prod, g in train.groupby("product"):
            g = g.sort_values("date")
            aux = g.rename(columns={"date":"ds", TARGET:"y"}).copy()
            m = Prophet(interval_width=0.80, weekly_seasonality=True,
                        daily_seasonality=False, yearly_seasonality=False)
            if PROPHET_USE_REGRESSORS and candidate_regressors:
                for reg in candidate_regressors:
                    m.add_regressor(reg)
            cols_fit = ["ds","y"] + (candidate_regressors if PROPHET_USE_REGRESSORS else [])
            aux[cols_fit] = aux[cols_fit].sort_values("ds").ffill().bfill()
            m.fit(aux[cols_fit])

            future = pd.DataFrame({"ds": pd.date_range(test_start, test_end, freq="D")})
            if PROPHET_USE_REGRESSORS and candidate_regressors:
                regs_future = (df[df["product"]==prod].rename(columns={"date":"ds"})[["ds"] + candidate_regressors]
                               .drop_duplicates(subset=["ds"]).sort_values("ds").ffill().bfill())
                future = future.merge(regs_future, on="ds", how="left")
                future[candidate_regressors] = future[candidate_regressors].ffill().bfill()

            fcst = m.predict(future)[["ds","yhat","yhat_lower","yhat_upper"]]
            fcst = fcst.rename(columns={"yhat_lower":"yhat_p10", "yhat_upper":"yhat_p90"})
            fcst["product"] = prod
            fold.append(fcst.rename(columns={"ds":"date"}))

        fold = pd.concat(fold, ignore_index=True) if fold else pd.DataFrame(columns=["date","yhat","yhat_p10","yhat_p90","product"])
        mask = (fold["date"] >= test_start) & (fold["date"] <= test_end)
        fold = fold.loc[mask].copy()
        fold["h"] = (fold["date"] - test_start).dt.days + 1

        y_true = test[["date","product",TARGET]].copy()
        merged = y_true.merge(fold, on=["date","product"], how="left")
        merged["model"] = "prophet"
        prophet_all.append(merged)

    prophet_results = (pd.concat(prophet_all, ignore_index=True)
                       if prophet_all else
                       pd.DataFrame(columns=["date","product",TARGET,"h","yhat","yhat_p10","yhat_p90","model"]))

    by_h_prophet, overall_prophet = summarize_metrics(prophet_results.rename(columns={TARGET: "y"}))

    if register_final:
        # Serie agregada por día (suma del TARGET)
        agg = df.groupby("date", as_index=False)[TARGET].sum().sort_values("date")
        agg = agg.rename(columns={"date": "ds", TARGET: "y"}).copy()

        m_final = Prophet(interval_width=0.80, weekly_seasonality=True,
                          daily_seasonality=False, yearly_seasonality=False)

        
        if PROPHET_USE_REGRESSORS and candidate_regressors:
            for reg in candidate_regressors:
                m_final.add_regressor(reg)
            regs_agg = (df.groupby("date", as_index=False)[candidate_regressors].mean()
                          .rename(columns={"date": "ds"}).sort_values("ds"))
            agg = agg.merge(regs_agg, on="ds", how="left")
            agg[candidate_regressors] = agg[candidate_regressors].ffill().bfill()

        m_final.fit(agg)

        metric_name = "sMAPE"
        best_score = float(overall_prophet.get(metric_name, 0.0)) if isinstance(overall_prophet, dict) else 0.0

        entry = register_best_model(
            model_obj=m_final,        # << Prophet serializado con joblib
            name="prophet",           
            metric=metric_name,
            score=best_score,
            horizon=HORIZON
        )
        print("[Prophet] best_model actualizado:", entry)
   

    return prophet_results, by_h_prophet, overall_prophet
