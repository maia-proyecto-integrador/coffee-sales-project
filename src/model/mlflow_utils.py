from pathlib import Path
import pandas as pd
import mlflow

def init_mlflow(results_dir: Path):
    mlruns = (results_dir / "mlruns").resolve()
    mlflow.set_tracking_uri(mlruns.as_uri())
    mlflow.set_experiment("CoffeeForecasting")
    return mlruns

def mlflow_log_run(model_name: str,
                   params: dict,
                   overall_csv: Path,
                   by_h_csv: Path,
                   extra_artifacts: list = None,
                   tags: dict = None):
    with mlflow.start_run(run_name=model_name):
        if params:
            clean = {}
            for k, v in params.items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    clean[k] = v
                else:
                    clean[k] = str(v)
            mlflow.log_params(clean)
        if tags: mlflow.set_tags(tags)

        # overall metrics
        try:
            if Path(overall_csv).exists():
                ovr = pd.read_csv(overall_csv)
                if not ovr.empty:
                    row = ovr.iloc[0].to_dict()
                    for key in ["MAE","RMSE","sMAPE","MAPE","COV_p10_p90_%"]:
                        if key in row and pd.notna(row[key]):
                            mlflow.log_metric(key, float(row[key]))
        except Exception as e:
            print(f"[mlflow] warning leyendo overall '{overall_csv}':", e)

        # by_h metrics
        try:
            if Path(by_h_csv).exists():
                byh = pd.read_csv(by_h_csv)
                if not byh.empty:
                    if "h" in byh.columns:
                        steps = byh["h"].astype(int).tolist()
                    else:
                        steps = list(range(1, len(byh)+1))
                    for i, (_, r) in enumerate(byh.iterrows()):
                        step = steps[i]
                        for key in ["MAE","RMSE","sMAPE","MAPE","COV_p10_p90_%"]:
                            if key in r and pd.notna(r[key]):
                                mlflow.log_metric(f"{key}_by_h", float(r[key]), step=int(step))
        except Exception as e:
            print(f"[mlflow] warning leyendo by_h '{by_h_csv}':", e)

        for p in [overall_csv, by_h_csv]:
            p = Path(p)
            if p.exists(): mlflow.log_artifact(str(p))
        if extra_artifacts:
            for p in extra_artifacts:
                p = Path(p)
                if p.exists(): mlflow.log_artifact(str(p))
