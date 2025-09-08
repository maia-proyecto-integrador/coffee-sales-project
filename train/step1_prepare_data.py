# ===========================
# step1_prepare_data.py
# ===========================
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_PATH = "data/processed/coffee_ml_features.csv"
INDEX_PATH = "data/clean/index_1.csv"
RESULTS_DIR = Path("results/artifacts")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "transactions"
HOLIDAY_COL = "is_holiday"

# ===========================
# 1. Cargar dataset base
# ===========================
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Derivar product desde columnas one-hot product_*
prod_cols = [c for c in df.columns if c.startswith("product_")]
if prod_cols:
    df["product"] = df[prod_cols].idxmax(axis=1).str.replace("product_", "")
else:
    assert "product" in df.columns, "Debe existir columna 'product'"

print("✅ Dataset cargado:", df.shape)

# ===========================
# 2. Merge con clima/festivos
# ===========================
def build_daily_from_index(index_path, holiday_col="is_holiday"):
    index_path = Path(index_path)
    if not index_path.exists():
        return None
    raw = pd.read_csv(index_path)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    agg_dict = {
        "wx_temperature_2m": "mean",
        "wx_precipitation": "sum",
        "wx_cloudcover": "mean"
    }
    if holiday_col in raw.columns:
        agg_dict[holiday_col] = "max"
    daily = raw.groupby("date").agg(agg_dict).reset_index()
    return daily

daily_idx = build_daily_from_index(INDEX_PATH, HOLIDAY_COL)
if daily_idx is not None:
    df = df.merge(daily_idx, on="date", how="left")
    print("✅ Merge con clima/festivos:", daily_idx.columns.tolist())

# ===========================
# 3. Reindexar calendario
# ===========================
def ensure_complete_calendar(dfin):
    out = []
    for prod, g in dfin.groupby("product"):
        g = g.sort_values("date")
        full_idx = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        g = g.set_index("date").reindex(full_idx)
        g["product"] = prod
        g.index.name = "date"
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)

df = ensure_complete_calendar(df)

# ===========================
# 4. Features calendario
# ===========================
dt = pd.to_datetime(df["date"])
df["dow"] = dt.dt.dayofweek
df["month"] = dt.dt.month
df["is_weekend"] = (df["dow"] >= 5).astype(int)
df["is_month_start"] = dt.dt.is_month_start.astype(int)
df["is_month_end"] = dt.dt.is_month_end.astype(int)
df["dow_sin"] = np.sin(2*np.pi*dt.dt.dayofweek/7)
df["dow_cos"] = np.cos(2*np.pi*dt.dt.dayofweek/7)
df["month_sin"] = np.sin(2*np.pi*(dt.dt.month-1)/12)
df["month_cos"] = np.cos(2*np.pi*(dt.dt.month-1)/12)

# ===========================
# 5. Features causales
# ===========================
def build_causal_features(dfin, ycol=TARGET):
    dfin = dfin.sort_values(["product", "date"]).copy()
    out = []
    for prod, g in dfin.groupby("product"):
        g = g.sort_values("date").copy()
        g[f"{ycol}_lag_1"] = g[ycol].shift(1)
        g[f"{ycol}_lag_7"] = g[ycol].shift(7)
        g[f"{ycol}_lag_14"] = g[ycol].shift(14)
        s = g[ycol].shift(1)
        g[f"{ycol}_roll_3"] = s.rolling(3, min_periods=1).mean()
        g[f"{ycol}_roll_7"] = s.rolling(7, min_periods=1).mean()
        g[f"{ycol}_roll_30"] = s.rolling(30, min_periods=1).mean()
        g[f"{ycol}_vol_7"] = s.rolling(7, min_periods=1).std()
        out.append(g)
    return pd.concat(out, ignore_index=True)

df = build_causal_features(df, ycol=TARGET)

# ===========================
# 6. Agregados de negocio
# ===========================
def add_business_aggregates(dfin, ycol=TARGET):
    dfo = dfin.sort_values(["date", "product"]).copy()
    totals = dfo.groupby("date")[ycol].sum().rename("totals_day")
    dfo = dfo.merge(totals, left_on="date", right_index=True, how="left")
    dfo["share_day"] = np.where(dfo["totals_day"] > 0, dfo[ycol]/dfo["totals_day"], 0.0)
    dfo["totals_day_t1"] = dfo.groupby("product")["totals_day"].shift(1)
    dfo["share_day_t1"] = dfo.groupby("product")["share_day"].shift(1)
    dfo["totals_day_roll_7"] = dfo.groupby("product")["totals_day_t1"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    dfo["competitor_sum_t1"] = dfo["totals_day_t1"] - dfo.groupby("product")[ycol].shift(1)
    dfo.drop(columns=["totals_day", "share_day"], inplace=True)
    return dfo

df = add_business_aggregates(df, ycol=TARGET)

# ===========================
# 7. Guardar resultados
# ===========================
schema = pd.Series({col: str(df[col].dtype) for col in df.columns})
schema.to_csv(RESULTS_DIR / "df_final_schema.csv", header=["dtype"])
df.to_csv(RESULTS_DIR / "df_final.csv", index=False)
df.to_parquet(RESULTS_DIR / "df_final.parquet", index=False)

print("✅ Preprocesamiento completo")
print(" - df_final.csv guardado en results/artifacts/")
print(" - df_final.parquet guardado en results/artifacts/")
print(" - df_final_schema.csv guardado en results/artifacts/")
