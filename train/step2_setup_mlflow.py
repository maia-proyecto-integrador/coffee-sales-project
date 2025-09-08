# ===========================
# 📦 Importaciones
# ===========================
import mlflow
from pathlib import Path
import numpy as np

# ===========================
# 📂 Paths
# ===========================
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Carpeta local donde MLflow guardará runs y artefactos
MLFLOW_DIR = (RESULTS_DIR / "mlruns").resolve()

# ===========================
# ⚙️ Configuración de MLflow
# ===========================
mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
mlflow.set_experiment("CoffeeForecasting")

print(f"✅ MLflow inicializado en {MLFLOW_DIR}")
print("   Experimento: CoffeeForecasting")

# ===========================
# 📊 Métricas personalizadas
# ===========================
def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))

def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2)))

def mape(y_true, y_pred, epsilon=1e-6):
    """MAPE seguro (ignora divisiones por 0)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = np.abs(y_true) > epsilon
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def smape(y_true, y_pred, epsilon=1e-6):
    """sMAPE (%)"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)).clip(min=epsilon)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100)

print("✅ Funciones de métricas listas: MAE, RMSE, MAPE, sMAPE")
