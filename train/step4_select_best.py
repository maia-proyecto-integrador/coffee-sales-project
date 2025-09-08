# ========================================
# 🏆 Selección del mejor modelo y registro
# ========================================

import pandas as pd
import numpy as np
from pathlib import Path
import json, joblib
from datetime import datetime

# Paths
RESULTS_DIR = Path("results")
MODELS_DIR = RESULTS_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_PATH = MODELS_DIR / "registry.json"

# Config
BEST_METRIC = "sMAPE"   # métrica a minimizar
EXCLUDE_TAGS = ("baseline", "baselines", "ensemble")

# 1) Recolectar métricas
overall_files = [
    f for f in RESULTS_DIR.glob("*_metrics_overall.csv")
    if not f.name.startswith("all_models")
]

if not overall_files:
    raise FileNotFoundError("No se encontraron archivos *_metrics_overall.csv en results/")

rows = []
for f in overall_files:
    try:
        df = pd.read_csv(f)
        if df.empty:
            continue
        model_name = f.stem.replace("_metrics_overall", "")
        df["model"] = model_name
        rows.append(df.iloc[0])
    except Exception as e:
        print(f"[WARN] Error leyendo {f}: {e}")

summary = pd.DataFrame(rows)
summary.to_csv(RESULTS_DIR / "all_models_metrics_overall.csv", index=False)
print("✅ Resumen de métricas guardado → all_models_metrics_overall.csv")

# 2) Seleccionar el mejor modelo
if BEST_METRIC not in summary.columns:
    raise ValueError(f"La métrica {BEST_METRIC} no está en summary. Columnas disponibles: {summary.columns.tolist()}")

summary[BEST_METRIC] = pd.to_numeric(summary[BEST_METRIC], errors="coerce")
candidatos = summary[
    ~summary["model"].str.lower().str.contains("|".join(EXCLUDE_TAGS))
].dropna(subset=[BEST_METRIC])

if candidatos.empty:
    raise ValueError("No hay modelos candidatos válidos para selección.")

best_row = candidatos.sort_values(BEST_METRIC, ascending=True).iloc[0]
BEST_MODEL_NAME = str(best_row["model"])
BEST_SCORE = float(best_row[BEST_METRIC])

print(f"🏆 Mejor modelo: {BEST_MODEL_NAME} con {BEST_METRIC}={BEST_SCORE:.4f}")

# 3) Guardar en registry.json
ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
registry = {
    "best_model": {
        "name": BEST_MODEL_NAME,
        "metric": BEST_METRIC,
        "score": BEST_SCORE,
        "horizon": 15,
        "trained_at": ts_tag
    }
}
REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False))
print(f"✅ Registry actualizado → {REGISTRY_PATH}")

# 4) Serialización por tipo de modelo
if BEST_MODEL_NAME.startswith("lgbm"):
    print("💾 Serializando LightGBM...")

    df = pd.read_csv(RESULTS_DIR / "artifacts" / "df_final.csv")
    from lightgbm import LGBMRegressor
    y = df["transactions"]
    X = df.drop(columns=["transactions", "date", "product"])

    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )
    model.fit(X.fillna(0), y)

    path = MODELS_DIR / f"{BEST_MODEL_NAME}_{ts_tag}.joblib"
    joblib.dump(model, path)

    registry["best_model"]["artifact"] = str(path)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False))

    print(f"✅ Modelo LGBM guardado → {path}")

elif BEST_MODEL_NAME.startswith("prophet"):
    print("💾 Serializando Prophet...")

    from prophet import Prophet

    df = pd.read_csv(RESULTS_DIR / "artifacts" / "df_final.csv", parse_dates=["date"])
    ts_total = df.groupby("date")["transactions"].sum().reset_index()
    ts_total.columns = ["ds", "y"]

    m = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False
    )
    m.fit(ts_total)

    path = MODELS_DIR / f"{BEST_MODEL_NAME}_{ts_tag}.joblib"
    joblib.dump(m, path)

    registry["best_model"]["artifact"] = str(path)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False))

    print(f"✅ Modelo Prophet guardado → {path}")

elif BEST_MODEL_NAME.startswith("sarimax"):
    print("💾 Serializando SARIMAX...")

    import statsmodels.api as sm

    df = pd.read_csv(RESULTS_DIR / "artifacts" / "df_final.csv", parse_dates=["date"])
    ts_total = df.groupby("date")["transactions"].sum().asfreq("D").fillna(0.0)

    model = sm.tsa.statespace.SARIMAX(
        ts_total,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res = model.fit(disp=False)

    path = MODELS_DIR / f"{BEST_MODEL_NAME}_{ts_tag}.joblib"
    joblib.dump(res, path)

    registry["best_model"]["artifact"] = str(path)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False))

    print(f"✅ Modelo SARIMAX guardado → {path}")

else:
    print(f"⚠️ {BEST_MODEL_NAME} aún no tiene bloque de serialización implementado.")
