from pathlib import Path
import pandas as pd

def normalize_for_forecast_save(df_in: pd.DataFrame, target: str) -> pd.DataFrame:
    df = df_in.copy()
    if "h" not in df.columns:
        if "index" in df.columns: df = df.rename(columns={"index": "h"})
        elif isinstance(df.index, pd.MultiIndex) and "h" in df.index.names: df = df.reset_index()
    y_candidates = ["y", target, "revenue", "units", "qty", "quantity", "sales", "target", "y_true"]
    y_col = next((c for c in y_candidates if c in df.columns), None)
    if y_col is not None and y_col != "y":
        df = df.rename(columns={y_col: "y"})
    cols = [c for c in ["date","product","h","y","yhat","yhat_p10","yhat_p90","model"] if c in df.columns]
    return df[cols]

def save_metrics_and_forecasts(name: str, results_df: pd.DataFrame, by_h_df: pd.DataFrame, overall: dict, results_dir: Path, target: str):
    results_dir.mkdir(exist_ok=True, parents=True)
    results_df.to_csv(results_dir / f"{name}_forecasts.csv", index=False)
    norm = normalize_for_forecast_save(results_df, target)
    norm.to_csv(results_dir / f"{name}_forecasts.csv", index=False)
    if by_h_df is not None:
        if "h" not in by_h_df.columns:
            by_h_df = by_h_df.reset_index()
            if "index" in by_h_df.columns and "h" not in by_h_df.columns:
                by_h_df = by_h_df.rename(columns={"index": "h"})
        by_h_df = by_h_df.sort_values("h").reset_index(drop=True)
        by_h_df.to_csv(results_dir / f"{name}_metrics_by_h.csv", index=False)
    pd.DataFrame([overall]).to_csv(results_dir / f"{name}_metrics_overall.csv", index=False)
    return (results_dir / f"{name}_metrics_overall.csv", results_dir / f"{name}_metrics_by_h.csv")
