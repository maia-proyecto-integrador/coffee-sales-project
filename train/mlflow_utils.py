import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from step2_setup_mlflow import rmse, mae, mape, smape

# Serie simulada
ts = pd.Series([100, 120, 130, 115, 140], 
               index=pd.date_range("2024-01-01", periods=5, freq="D"))

# Entrenar Holt-Winters
model = ExponentialSmoothing(ts, trend="add", seasonal="add", seasonal_periods=2)
fit = model.fit(optimized=True)
forecast = fit.forecast(3)

# DataFrame forecast
fc_df = pd.DataFrame({
    "date": forecast.index,
    "real": [None, None, None],  # todavía no hay test
    "forecast": forecast.values
})

# Métricas (ejemplo, aquí solo sobre train)
metrics = {
    "RMSE": rmse(ts, fit.fittedvalues),
    "MAE": mae(ts, fit.fittedvalues),
    "MAPE": mape(ts, fit.fittedvalues),
    "sMAPE": smape(ts, fit.fittedvalues),
}

# Registrar corrida en MLflow
# train/mlflow_utils.py

import mlflow
import pandas as pd
from pathlib import Path

def _mlflow_log_run(
    model_name: str,
    params: dict,
    overall_csv: Path,
    by_h_csv: Path,
    extra_artifacts: list = None,
    tags: dict = None
):
    """Registra parámetros, métricas y artefactos en MLflow."""
    with mlflow.start_run(run_name=model_name):
        # parámetros
        if params:
            clean = {k: (str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v)
                     for k, v in params.items()}
            mlflow.log_params(clean)

        if tags:
            mlflow.set_tags(tags)

        # métricas overall
        if Path(overall_csv).exists():
            ovr = pd.read_csv(overall_csv)
            if not ovr.empty:
                row = ovr.iloc[0].to_dict()
                for key in ["MAE","RMSE","sMAPE","MAPE","COV_p10_p90_%"]:
                    if key in row and pd.notna(row[key]):
                        mlflow.log_metric(key, float(row[key]))

        # métricas por horizonte
        if Path(by_h_csv).exists():
            byh = pd.read_csv(by_h_csv)
            if not byh.empty:
                steps = byh["h"].astype(int).tolist() if "h" in byh.columns else list(range(1, len(byh)+1))
                for i, (_, r) in enumerate(byh.iterrows()):
                    for key in ["MAE","RMSE","sMAPE","MAPE","COV_p10_p90_%"]:
                        if key in r and pd.notna(r[key]):
                            mlflow.log_metric(f"{key}_by_h", float(r[key]), step=int(steps[i]))

        # artefactos
        for p in [overall_csv, by_h_csv] + (extra_artifacts or []):
            p = Path(p)
            if p.exists():
                mlflow.log_artifact(str(p))
