from pathlib import Path
from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel
from strictyaml import YAML, load

import os

# Project Directories
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained"

# Ensure Keras backend as Torch (as in the notebook/script)
os.environ.setdefault("KERAS_BACKEND", "torch")  # must be set before importing keras

# ===========================
# 🎯 PARÁMETROS GENERALES
# ===========================
TARGET = "transactions"           # Variable objetivo para todos los modelos
HORIZON = 7                       # Días a pronosticar (todos los modelos)
N_ORIGINS = 4                     # Número de orígenes para backtesting (todos los modelos)
MIN_TRAIN_DAYS = 150              # Días mínimos de entrenamiento (todos los modelos)
TOPK_IMP = 40                     # Top K features por importancia (LGBM principalmente)

# ===========================
# 📊 PREPROCESAMIENTO DE DATOS
# ===========================
USE_LOG1P_TARGET = False          # Transformación log1p del target (todos los modelos)
CAP_OUTLIERS = False              # Cap outliers en el target (todos los modelos)
OUTLIER_Q = 0.995                 # Quantil para cap de outliers (todos los modelos)

# ===========================
# 📅 FEATURES DE CALENDARIO
# ===========================
USE_RICH_CALENDAR = True          # Usar features de calendario enriquecidas (todos los modelos)
UA_HOLIDAYS_PATH = None           # Ruta a archivo de festivos de Ucrania (todos los modelos)
ADD_BUSINESS_AGGREGATES = True    # Agregar agregados de negocio (todos los modelos)

# ===========================
# 🌤️ FEATURES EXTERNAS (CLIMA/FESTIVOS)
# ===========================
USE_INDEX_WEATHER_HOLIDAYS = True # Usar datos de clima y festivos (SARIMAX, Prophet)
WEATHER_AGG = {                   # Agregaciones de variables climáticas (SARIMAX, Prophet)
    "wx_temperature_2m": "mean",   # Temperatura promedio diaria
    "wx_precipitation": "sum",     # Precipitación total diaria
    "wx_cloudcover": "mean"        # Cobertura de nubes promedio
}
HOLIDAY_COL = "is_holiday"        # Columna de festivos (SARIMAX, Prophet)

# ===========================
# 🎲 PARÁMETROS DE REPRODUCIBILIDAD
# ===========================
RANDOM_STATE = 50                 # Semilla para reproducibilidad (todos los modelos)

class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    data_file: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    target: str
    features: List[str]
    test_size: float
    random_state: int
    n_estimators: int
    max_depth: int
    temp_features: List[str]
    qual_vars: List[str]
    categorical_vars: Sequence[str]
    qual_mappings: Dict[str, int]

class Config(BaseModel):
    """Master config object."""

    app_config: AppConfig
    model_config_: ModelConfig

def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")

def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")

def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # specify the data attribute from the strictyaml YAML type.
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        model_config_=ModelConfig(**parsed_config.data),
    )

    return _config

config = create_and_validate_config()
