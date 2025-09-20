from pathlib import Path
import numpy as np
import pandas as pd
from .config import (TARGET, CAP_OUTLIERS, OUTLIER_Q, USE_INDEX_WEATHER_HOLIDAYS,
                     HOLIDAY_COL, WEATHER_AGG, USE_RICH_CALENDAR, UA_HOLIDAYS_PATH,
                     ADD_BUSINESS_AGGREGATES)
from typing import Tuple

def load_features(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    if "date" not in df.columns:
        raise AssertionError("Se requiere columna 'date'")
    df["date"] = pd.to_datetime(df["date"])
    prod_cols = [c for c in df.columns if c.startswith("product_")]
    if len(prod_cols) == 0:
        raise ValueError("No hay columnas product_* (one-hot).")
    df["product"] = df[prod_cols].idxmax(axis=1).str.replace("product_", "", regex=False)
    return df

def build_daily_from_index(index_path: Path, holiday_col="is_holiday", weather_agg=None):
    index_path = Path(index_path)
    if not index_path.exists():
        print("[Aviso] INDEX_PATH no existe; se omite merge clima/festivo.")
        return None
    raw = pd.read_csv(index_path)
    if "date" not in raw.columns:
        raise ValueError("index_1.csv debe tener columna 'date'.")
    raw["date"] = pd.to_datetime(raw["date"])
    agg_dict = {}
    if weather_agg:
        for k, v in weather_agg.items():
            if k in raw.columns:
                agg_dict[k] = v
    if holiday_col in raw.columns:
        agg_dict[holiday_col] = "max"
    if not agg_dict:
        print("[Aviso] No se encontraron columnas de clima/festivo en index_1 para agregar.")
        return None
    daily = raw.groupby("date").agg(agg_dict).reset_index()
    return daily

def ensure_complete_calendar(dfin: pd.DataFrame) -> pd.DataFrame:
    out = []
    for prod, g in dfin.groupby("product"):
        g = g.sort_values("date")
        full_idx = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        g = g.set_index("date").reindex(full_idx)
        g["product"] = prod
        g.index.name = "date"
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)

def rebuild_causal_rollings(dfin: pd.DataFrame, ycol=TARGET):
    dfin = dfin.sort_values(["product","date"]).copy()
    cols = [c for c in dfin.columns if c.startswith(f"{ycol}_roll_")]
    if not cols:
        return dfin, pd.DataFrame(columns=["column","window","reconstructed"])
    log = []
    out = []
    for prod, g in dfin.groupby("product"):
        g = g.sort_values("date").copy()
        s = g[ycol].shift(1)  # causal
        for rc in cols:
            try:
                w = int(rc.split("_")[-1])
            except:
                w = None
            if w is None: 
                continue
            g[rc] = s.rolling(w, min_periods=1).mean()
            log.append({"column": rc, "window": w, "reconstructed": True})
        out.append(g)
    log_df = pd.DataFrame(log).drop_duplicates().sort_values(["window","column"])
    return pd.concat(out, ignore_index=True), log_df

def add_calendar_rich(dfin: pd.DataFrame, holidays_path=None):
    dfo = dfin.copy()
    dt = pd.to_datetime(dfo["date"])
    dfo["dow"] = dt.dt.dayofweek
    dfo["month"] = dt.dt.month
    dfo["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    dfo["is_weekend"] = (dfo["dow"] >= 5).astype(int)
    dfo["is_month_start"] = dt.dt.is_month_start.astype(int)
    dfo["is_month_end"] = dt.dt.is_month_end.astype(int)
    if "is_holiday" not in dfo.columns:
        dfo["is_holiday"] = 0
    dfo = dfo.sort_values("date")
    dfo["is_holiday_prev"] = dfo["is_holiday"].shift(1).fillna(0).astype(int)
    dfo["is_holiday_next"] = dfo["is_holiday"].shift(-1).fillna(0).astype(int)
    dfo["dow_sin"] = np.sin(2*np.pi*dt.dt.dayofweek/7)
    dfo["dow_cos"] = np.cos(2*np.pi*dt.dt.dayofweek/7)
    dfo["month_sin"] = np.sin(2*np.pi*(dt.dt.month-1)/12)
    dfo["month_cos"] = np.cos(2*np.pi*(dt.dt.month-1)/12)
    if holidays_path:
        h = pd.read_csv(holidays_path)
        h["date"] = pd.to_datetime(h["date"])
        h["is_holiday_ext"] = 1
        dfo = dfo.merge(h[["date","is_holiday_ext"]], on="date", how="left")
        dfo["is_holiday_ext"] = dfo["is_holiday_ext"].fillna(0).astype(int)
    return dfo

def add_business_aggregates_tminus1(dfin: pd.DataFrame, ycol=TARGET):
    dfo = dfin.sort_values(["date","product"]).copy()
    totals = dfo.groupby("date")[ycol].sum().rename("totals_day")
    dfo = dfo.merge(totals, left_on="date", right_index=True, how="left")
    dfo["share_day"] = (dfo[ycol] / dfo["totals_day"]).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    dfo["totals_day_t1"] = dfo.groupby("product")["totals_day"].shift(1)
    dfo["share_day_t1"]  = dfo.groupby("product")["share_day"].shift(1)
    dfo = dfo.sort_values(["product","date"])
    dfo["totals_day_roll_7"] = dfo.groupby("product")["totals_day_t1"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    dfo["competitor_sum_t1"] = dfo["totals_day_t1"] - dfo.groupby("product")[ycol].shift(1)
    dfo.drop(columns=["totals_day","share_day"], inplace=True)
    return dfo

def build_causal_features(dfin: pd.DataFrame, ycol=TARGET):
    dfin = dfin.sort_values(["product","date"]).copy()
    out = []
    for prod, g in dfin.groupby("product"):
        g = g.sort_values("date").copy()
        g[f"{ycol}_lag_1"]  = g[ycol].shift(1)
        g[f"{ycol}_lag_7"]  = g[ycol].shift(7)
        g[f"{ycol}_lag_14"] = g[ycol].shift(14)
        s = g[ycol].shift(1)
        g[f"{ycol}_roll_3"]  = s.rolling(3,  min_periods=1).mean()
        g[f"{ycol}_roll_7"]  = s.rolling(7,  min_periods=1).mean()
        g[f"{ycol}_roll_30"] = s.rolling(30, min_periods=1).mean()
        g[f"{ycol}_vol_7"]   = s.rolling(7,  min_periods=1).std()
        out.append(g)
    return pd.concat(out, ignore_index=True)

def cap_outliers_if_needed(df, ycol=TARGET, enabled=False, q=0.995):
    if not enabled: return df
    q_map = df.groupby("product")[ycol].quantile(q).rename("q_hi")
    df = df.merge(q_map, on="product", how="left")
    df[ycol] = np.where(df[ycol] > df["q_hi"], df["q_hi"], df[ycol])
    return df.drop(columns=["q_hi"])

def persist_snapshot(df_final: pd.DataFrame, results_dir: Path):
    art_dir = (results_dir / "artifacts"); art_dir.mkdir(parents=True, exist_ok=True)
    df_final.sample(min(1000, len(df_final))).to_csv(art_dir / "df_final_sample.csv", index=False)
    df_final.to_parquet(art_dir / "df_final.parquet")
    import pandas as pd
    pd.DataFrame({"column": df_final.columns, "dtype": [str(t) for t in df_final.dtypes]}).to_csv(art_dir / "df_final_schema.csv", index=False)
    return art_dir
