

# utils/coffee_eda_utils.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict
import pandas as pd

# --------- Config por defecto (borra si ya lo tienes en tu proyecto) ----------
@dataclass
class ColumnsConfig:
    datetime_col: str = "datetime"
    amount_col:   str = "money"
    product_col:  str = "coffee_name"
    payment_col:  str = "cash_type"
    machine_id_col: Optional[str] = None
    store_id_col:   Optional[str] = None
    category_col:   Optional[str] = None

DEFAULT_COLS = ColumnsConfig()

# --------- Helpers de tipado / limpieza --------------------------------------
def standardize_types(df: pd.DataFrame, cols: ColumnsConfig = DEFAULT_COLS) -> pd.DataFrame:
    """Castea tipos básicos según el esquema. No renombra columnas."""
    d = df.copy()
    if cols.product_col in d.columns:
        d[cols.product_col] = d[cols.product_col].astype("string")
    if cols.payment_col and cols.payment_col in d.columns:
        d[cols.payment_col] = d[cols.payment_col].astype("string")
    if cols.amount_col in d.columns:
        d[cols.amount_col] = pd.to_numeric(d[cols.amount_col], errors="coerce")
    for opt in (cols.machine_id_col, cols.store_id_col, cols.category_col):
        if opt and opt in d.columns:
            d[opt] = d[opt].astype("string")
    return d

def drop_strict_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina duplicados exactos y loguea cuántos se fueron."""
    before = len(df)
    d = df.drop_duplicates()
    # print(f"Dropped {before - len(d)} exact duplicates.")
    return d

def ensure_dt_local(
    df: pd.DataFrame,
    dt_col: str = "datetime",
    rename_to: Optional[str] = None
) -> pd.DataFrame:
    """
    Asegura que dt_col sea datetime (naive o localized). Si rename_to se pasa, renombra.
    Por defecto mantiene el nombre original (recomendado).
    """
    d = df.copy()
    d[dt_col] = pd.to_datetime(d[dt_col], errors="coerce")
    d = d.dropna(subset=[dt_col]).sort_values(dt_col)
    if rename_to and rename_to != dt_col:
        d = d.rename(columns={dt_col: rename_to})
    return d

def ensure_sorted(df: pd.DataFrame, cols: ColumnsConfig = DEFAULT_COLS) -> pd.DataFrame:
    """Ordena por la columna de fecha definida en cols.datetime_col."""
    return df.sort_values(by=[cols.datetime_col]).reset_index(drop=True)

# --------- Features de tiempo -------------------------------------------------
def add_time_features(df: pd.DataFrame, cols: ColumnsConfig = DEFAULT_COLS) -> pd.DataFrame:
    """Añade columnas: hour, dow (nombre), week, month (Period)."""
    d = df.copy()
    dt = pd.to_datetime(d[cols.datetime_col], errors="coerce")
    d = d.loc[dt.notna()].copy()
    dt = dt[dt.notna()]
    d["hour"] = dt.dt.hour
    d["dow"]  = dt.dt.day_name()
    d["week"] = dt.dt.to_period("W")
    d["month"]= dt.dt.to_period("M")
    return d

# --------- Utilidades de análisis --------------------------------------------
def pick_y_col(df: pd.DataFrame, cols: ColumnsConfig = DEFAULT_COLS) -> str:
    """Devuelve la columna numérica objetivo (por defecto money)."""
    y = cols.amount_col
    if y in df.columns and pd.api.types.is_numeric_dtype(df[y]):
        return y
    raise KeyError(f"No numeric target found. Expected: {y}")

def pareto_series(df: pd.DataFrame, cols: ColumnsConfig = DEFAULT_COLS) -> pd.Series:
    """Serie de revenue por producto (ordenada desc)."""
    y = pick_y_col(df, cols)
    return df.groupby(cols.product_col)[y].sum().sort_values(ascending=False)

def calendar_daily_series(df: pd.DataFrame, cols: ColumnsConfig = DEFAULT_COLS) -> pd.Series:
    """
    Serie diaria con DatetimeIndex (para calmap):
    suma por día, resamplada a 'D' para rellenar huecos.
    """
    d = df.copy()
    d[cols.datetime_col] = pd.to_datetime(d[cols.datetime_col], errors="coerce")
    d = d.dropna(subset=[cols.datetime_col])
    s = d.set_index(cols.datetime_col)[cols.amount_col].resample("D").sum()
    return s
