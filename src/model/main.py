from pathlib import Path
import pandas as pd
import sys
from .config import *
from .paths import resolve_paths
from .mlflow_utils import init_mlflow, mlflow_log_run
from .data import (load_features, build_daily_from_index, ensure_complete_calendar,
                   rebuild_causal_rollings, add_calendar_rich, add_business_aggregates_tminus1,
                   build_causal_features, cap_outliers_if_needed, persist_snapshot)
from .results import save_metrics_and_forecasts
from .baselines import run_baselines
from .lgbm_direct import run_lgbm
from .prophet_model import run_prophet
from .sarimax_model import run_sarimax
from .lstm_direct import run_lstm_direct

def run_all(cwd: Path = None):
    cwd = Path.cwd() if cwd is None else Path(cwd)
    data_dir, results_dir, data_path, index_path = resolve_paths(cwd)

    print("=== RUTAS CLAVE ===")
    print("CWD:               ", cwd.resolve())
    print("REPO_ROOT (aprox): ", data_dir.parent.resolve())
    print("DATA_DIR:          ", data_dir.resolve())
    print("RESULTS_DIR:       ", results_dir.resolve())
    print("DATA_PATH:         ", data_path.resolve(), "| existe:", data_path.exists())
    print("INDEX_PATH:        ", index_path.resolve(), "| existe:", index_path.exists())

    if not data_path.exists():
        print("[AVISO] No se encontró DATA_PATH. Verifica 'data/processed/coffee_ml_features.csv'.")

    mlruns = init_mlflow(results_dir)

    # Load & enrich
    df = load_features(data_path)
    if USE_INDEX_WEATHER_HOLIDAYS:
        daily_idx = build_daily_from_index(index_path, HOLIDAY_COL, WEATHER_AGG)
        if daily_idx is not None:
            print("Agregados diarios desde index_1:", daily_idx.columns.tolist())
            df = df.merge(daily_idx, on="date", how="left")
        else:
            print("Sin merge desde index_1 (no disponible o sin columnas útiles).")

    # Check duplicates and reindex
    df = ensure_complete_calendar(df)
    df, _log = rebuild_causal_rollings(df, ycol=TARGET)
    df = cap_outliers_if_needed(df, ycol=TARGET, enabled=CAP_OUTLIERS, q=OUTLIER_Q)
    if USE_RICH_CALENDAR:
        df = add_calendar_rich(df, UA_HOLIDAYS_PATH)
    if ADD_BUSINESS_AGGREGATES:
        df = add_business_aggregates_tminus1(df, ycol=TARGET)
    df = build_causal_features(df, ycol=TARGET)

    # Persist snapshot
    art_dir = persist_snapshot(df.copy(), results_dir)

    # Run LGBM
    lgbm_results, by_h_lgbm, overall_lgbm, _ = run_lgbm(df.copy())
    ovr, byh = save_metrics_and_forecasts("lgbm_direct", lgbm_results, by_h_lgbm, overall_lgbm, results_dir, TARGET)
    mlflow_log_run("lgbm_direct",
                   params={"HORIZON": HORIZON, "N_ORIGINS": N_ORIGINS, "USE_LOG1P_TARGET": USE_LOG1P_TARGET},
                   overall_csv=ovr, by_h_csv=byh,
                   extra_artifacts=[results_dir / "lgbm_direct_forecasts.csv"], tags={"family": "lgbm", "stage": "train"})

    # Run Prophet
    try:
        prophet_results, by_h_prophet, overall_prophet = run_prophet(df.copy())
        ovr, byh = save_metrics_and_forecasts("prophet", prophet_results, by_h_prophet, overall_prophet, results_dir, TARGET)
        mlflow_log_run("prophet",
                       params={"HORIZON": HORIZON, "use_regressors": bool(PROPHET_USE_REGRESSORS)},
                       overall_csv=ovr, by_h_csv=byh,
                       extra_artifacts=[results_dir / "prophet_forecasts.csv"], tags={"family": "prophet", "stage": "train"})
    except Exception as e:
        print("Prophet no disponible/omitido:", e)

    # Run SARIMAX
    try:
        sarimax_results, by_h_sarimax, overall_sarimax = run_sarimax(df.copy())
        if not sarimax_results.empty:
            ovr, byh = save_metrics_and_forecasts("sarimax", sarimax_results, by_h_sarimax, overall_sarimax, results_dir, TARGET)
            mlflow_log_run("sarimax",
                           params={"HORIZON": HORIZON, "orders_searched": "[(1,0,1)x(1,0,1,7), (0,1,1)x(0,1,1,7)]"},
                           overall_csv=ovr, by_h_csv=byh,
                           extra_artifacts=[results_dir / "sarimax_forecasts.csv"], tags={"family": "sarimax", "stage": "train"})
    except Exception as e:
        print("SARIMAX no disponible/omitido:", e)

    # Run LSTM
    try:
        lstm_results, by_h_lstm, overall_lstm = run_lstm_direct(df.copy())
        ovr, byh = save_metrics_and_forecasts("lstm_direct", lstm_results, by_h_lstm, overall_lstm, results_dir, TARGET)
        mlflow_log_run("lstm_direct",
                       params={"HORIZON": HORIZON, "LOOKBACK": LSTM_LOOKBACK, "UNITS": LSTM_UNITS,
                               "DROPOUT": LSTM_DROPOUT, "LR": LSTM_LR, "BATCH": LSTM_BATCH,
                               "USE_LOG1P_TARGET": USE_LOG1P_TARGET},
                       overall_csv=ovr, by_h_csv=byh,
                       extra_artifacts=[results_dir / "lstm_direct_forecasts.csv"], tags={"family": "lstm", "stage": "train"})
    except Exception as e:
        print("LSTM no disponible/omitido:", e)

    # Baselines
    baseline_results, by_h_baseline, overall_baseline = run_baselines(df.copy(), results_dir)
    ovr, byh = save_metrics_and_forecasts("baselines", baseline_results, by_h_baseline, overall_baseline, results_dir, TARGET)
    mlflow_log_run("baselines",
                   params={"names": "naive1,snaive7,ma7", "HORIZON": HORIZON},
                   overall_csv=ovr, by_h_csv=byh,
                   extra_artifacts=[results_dir / "baselines_forecasts.csv"], tags={"family": "baselines", "stage": "train"})

    print("\nListo. Revisa resultados en:", results_dir)

if __name__ == "__main__":
    run_all()


def train_and_version(cwd: Path = None):
    cwd = Path.cwd() if cwd is None else Path(cwd)
    # 1) Data pipeline
    from .pipelines import data_pipeline, train_final_lgbm_direct, train_final_prophet, train_final_sarimax, train_final_lstm_direct
    df, data_dir, results_dir, data_path, index_path = data_pipeline(cwd)
    print("[Pipeline] Data listo para entrenamiento final.")
    # 2) Entrenamientos finales (serialización + versionado)
    artifacts = []
    try:
        artifacts.append(train_final_lgbm_direct(df.copy(), results_dir))
    except Exception as e:
        print("[Pipeline] LGBM final falló/omitido:", e)
    try:
        artifacts.append(train_final_prophet(df.copy(), results_dir))
    except Exception as e:
        print("[Pipeline] Prophet final falló/omitido:", e)
    try:
        artifacts.append(train_final_sarimax(df.copy(), results_dir))
    except Exception as e:
        print("[Pipeline] SARIMAX final falló/omitido:", e)
    try:
        artifacts.append(train_final_lstm_direct(df.copy(), results_dir))
    except Exception as e:
        print("[Pipeline] LSTM final falló/omitido:", e)
    print("\n[Pipeline] Artefactos versionados:")
    for a in artifacts:
        if a:
            print(" -", a["name"], "=>", a["version"], "|", a["path"])
    return artifacts
