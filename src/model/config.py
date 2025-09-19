import os
from pathlib import Path

# Ensure Keras backend as Torch (as in the notebook/script)
os.environ.setdefault("KERAS_BACKEND", "torch")  # must be set before importing keras

# -------------------------
# General parameters
# -------------------------
TARGET = "transactions"
HORIZON = 7
N_ORIGINS = 4
MIN_TRAIN_DAYS = 150
TOPK_IMP = 40

USE_LOG1P_TARGET = False
LGBM_OBJECTIVE = "auto"   # "auto" | "poisson" | "tweedie"
TWEEDIE_POWERS = [1.1, 1.3, 1.5]
CAP_OUTLIERS = False
OUTLIER_Q = 0.995

USE_RICH_CALENDAR = True
UA_HOLIDAYS_PATH = None
ADD_BUSINESS_AGGREGATES = True

USE_INDEX_WEATHER_HOLIDAYS = True
WEATHER_AGG = {
    "wx_temperature_2m": "mean",
    "wx_precipitation": "sum",
    "wx_cloudcover": "mean"
}
HOLIDAY_COL = "is_holiday"

PROPHET_USE_REGRESSORS = True

RANDOM_STATE = 42

# LSTM hyperparameters
LSTM_LOOKBACK = 30
LSTM_UNITS    = 64
LSTM_DROPOUT  = 0.2
LSTM_EPOCHS   = 50
LSTM_BATCH    = 256
LSTM_LR       = 1e-3
LSTM_PATIENCE = 5
