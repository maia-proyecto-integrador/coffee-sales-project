from pathlib import Path
import pandas as pd
import mlflow
import os

def init_mlflow(results_dir: Path):
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        # Caso 1: alguien definió un URI externo → usarlo
        mlflow.set_tracking_uri(uri)
    else:
        # Caso 2: por defecto, usar la carpeta local
        mlruns = (results_dir / "mlruns").resolve()
        mlflow.set_tracking_uri(mlruns.as_uri())

    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "CoffeeForecasting"))


def _mlflow_key(k: str) -> str:
    """Sanitiza nombres de métricas/params para que sean válidos en MLflow."""
    bad = {
        '%': 'pct',
        ' ': '_',
        '(': '_',
        ')': '_',
        ':': '_',
        ',': '_'
    }
    for a, b in bad.items():
        k = k.replace(a, b)
    return k

# lista única de métricas que usamos
METRICS = ["MAE", "RMSE", "sMAPE", "MAPE", "COV_p10_p90_%"]

def mlflow_log_run(model_name: str,
                   params: dict,
                   overall_csv: Path,
                   by_h_csv: Path,
                   extra_artifacts: list = None,
                   tags: dict = None):
    with mlflow.start_run(run_name=model_name):
        # parámetros
        if params:
            clean = {}
            for k, v in params.items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    clean[k] = v
                else:
                    clean[k] = str(v)
            mlflow.log_params(clean)

        # tags forzados a str
        if tags:
            mlflow.set_tags({str(k): str(v) for k, v in tags.items()})

        # métricas globales (overall)
        try:
            if Path(overall_csv).exists():
                ovr = pd.read_csv(overall_csv)
                if not ovr.empty:
                    row = ovr.iloc[0].to_dict()
                    for key in METRICS:
                        if key in row and pd.notna(row[key]):
                            val = float(row[key])
                            mlflow.log_metric(_mlflow_key(key), round(val, 6))
        except Exception as e:
            print(f"[mlflow] warning leyendo overall '{overall_csv}':", e)

        # métricas por horizonte (by_h)
        try:
            if Path(by_h_csv).exists():
                byh = pd.read_csv(by_h_csv)
                if not byh.empty:
                    if "h" in byh.columns:
                        steps = byh["h"].astype(int).tolist()
                    else:
                        steps = list(range(1, len(byh) + 1))
                    for i, (_, r) in enumerate(byh.iterrows()):
                        step = steps[i]
                        for key in METRICS:
                            if key in r and pd.notna(r[key]):
                                val = float(r[key])
                                mlflow.log_metric(
                                    _mlflow_key(f"{key}_by_h"),
                                    round(val, 6),
                                    step=int(step)
                                )
        except Exception as e:
            print(f"[mlflow] warning leyendo by_h '{by_h_csv}':", e)

        # log artefactos principales
        for p in [overall_csv, by_h_csv]:
            p = Path(p)
            if p.exists():
                mlflow.log_artifact(str(p))

        # artefactos extra (archivos o carpetas)
        if extra_artifacts:
            for p in extra_artifacts:
                p = Path(p)
                if p.is_dir():
                    mlflow.log_artifacts(str(p))
                elif p.exists():
                    mlflow.log_artifact(str(p))
