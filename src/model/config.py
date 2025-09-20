import os
from pathlib import Path

# Ensure Keras backend as Torch (as in the notebook/script)
os.environ.setdefault("KERAS_BACKEND", "torch")  # must be set before importing keras

# ===========================
# 🎯 PARÁMETROS GENERALES
# ===========================
# (Aplican a todos los modelos)

TARGET = "transactions"           # Variable objetivo para todos los modelos
HORIZON = 7                       # Días a pronosticar (todos los modelos)
N_ORIGINS = 4                     # Número de orígenes para backtesting (todos los modelos)
MIN_TRAIN_DAYS = 150              # Días mínimos de entrenamiento (todos los modelos)
TOPK_IMP = 40                     # Top K features por importancia (LGBM principalmente)

# ===========================
# 📊 PREPROCESAMIENTO DE DATOS
# ===========================
# (Aplican a todos los modelos)

USE_LOG1P_TARGET = False          # Transformación log1p del target (todos los modelos)
CAP_OUTLIERS = False              # Cap outliers en el target (todos los modelos)
OUTLIER_Q = 0.995                 # Quantil para cap de outliers (todos los modelos)

# ===========================
# 📅 FEATURES DE CALENDARIO
# ===========================
# (Aplican a todos los modelos)

USE_RICH_CALENDAR = True          # Usar features de calendario enriquecidas (todos los modelos)
UA_HOLIDAYS_PATH = None           # Ruta a archivo de festivos de Ucrania (todos los modelos)
ADD_BUSINESS_AGGREGATES = True    # Agregar agregados de negocio (todos los modelos)

# ===========================
# 🌤️ FEATURES EXTERNAS (CLIMA/FESTIVOS)
# ===========================
# (Aplican a SARIMAX y Prophet principalmente)

USE_INDEX_WEATHER_HOLIDAYS = True # Usar datos de clima y festivos (SARIMAX, Prophet)
WEATHER_AGG = {                   # Agregaciones de variables climáticas (SARIMAX, Prophet)
    "wx_temperature_2m": "mean",   # Temperatura promedio diaria
    "wx_precipitation": "sum",     # Precipitación total diaria
    "wx_cloudcover": "mean"        # Cobertura de nubes promedio
}
HOLIDAY_COL = "is_holiday"        # Columna de festivos (SARIMAX, Prophet)

# ===========================
# 🚀 LIGHTGBM PARÁMETROS
# ===========================

LGBM_OBJECTIVE = "auto"           # Objetivo de LightGBM: "auto" | "poisson" | "tweedie"
TWEEDIE_POWERS = [1.1, 1.3, 1.5] # Potencias Tweedie para probar (LGBM con objetivo tweedie)

# ===========================
# 📈 PROPHET PARÁMETROS
# ===========================

PROPHET_USE_REGRESSORS = True     # Usar regresores externos en Prophet (clima, festivos)

# ===========================
# 🧠 LSTM PARÁMETROS
# ===========================

LSTM_LOOKBACK = 30                # Ventana de lookback para LSTM (días hacia atrás)
LSTM_UNITS    = 64                # Número de unidades en capa LSTM
LSTM_DROPOUT  = 0.2               # Tasa de dropout para regularización
LSTM_EPOCHS   = 50                # Número máximo de épocas de entrenamiento
LSTM_BATCH    = 256               # Tamaño del batch para entrenamiento
LSTM_LR       = 1e-3              # Learning rate para optimizador
LSTM_PATIENCE = 5                 # Paciencia para early stopping

# ===========================
# 🎲 PARÁMETROS DE REPRODUCIBILIDAD
# ===========================
# (Aplican a todos los modelos)

RANDOM_STATE = 42                 # Semilla para reproducibilidad (todos los modelos)
