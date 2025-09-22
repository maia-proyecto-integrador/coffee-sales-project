from .config.core import (
    USE_INDEX_WEATHER_HOLIDAYS,
    HOLIDAY_COL,
    WEATHER_AGG,
    USE_RICH_CALENDAR,
    UA_HOLIDAYS_PATH,
    ADD_BUSINESS_AGGREGATES,
    TARGET,
    HORIZON,
    CAP_OUTLIERS,
    OUTLIER_Q,
)
from .pipeline import run_sarimax

from pathlib import Path
from .paths import resolve_paths
from .mlflow_utils import init_mlflow, mlflow_log_run
from .data import (
    load_features,
    build_daily_from_index,
    ensure_complete_calendar,
    rebuild_causal_rollings,
    add_calendar_rich,
    add_business_aggregates_tminus1,
    build_causal_features,
    cap_outliers_if_needed,
    persist_snapshot,
)
from .results import save_metrics_and_forecasts


def run_training(cwd: Path = None):
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
        print(
            "[AVISO] No se encontró DATA_PATH. Verifica 'datasets/coffee_ml_features.csv'."
        )

    init_mlflow(results_dir)

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
    persist_snapshot(df.copy(), results_dir)

    # Run SARIMAX
    try:
        sarimax_results, by_h_sarimax, overall_sarimax = run_sarimax(df.copy())
        if not sarimax_results.empty:
            ovr, byh = save_metrics_and_forecasts(
                "sarimax",
                sarimax_results,
                by_h_sarimax,
                overall_sarimax,
                results_dir,
                TARGET,
            )
            mlflow_log_run(
                "sarimax",
                params={
                    "HORIZON": HORIZON,
                    "orders_searched": "[(1,0,1)x(1,0,1,7), (0,1,1)x(0,1,1,7)]",
                },
                overall_csv=ovr,
                by_h_csv=byh,
                extra_artifacts=[results_dir / "sarimax_forecasts.csv"],
                tags={"family": "sarimax", "stage": "train"},
            )
    except Exception as e:
        print("SARIMAX no disponible/omitido:", e)

    print("\nListo. Revisa resultados en:", results_dir)


if __name__ == "__main__":
    run_training()
