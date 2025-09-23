import typing as t
import pandas as pd

from . import __version__ as _version
from .processing.data_manager import load_pipeline

pipeline_file_name = "sarimax.joblib"
_sarimax_model = load_pipeline(file_name=pipeline_file_name)


def make_prediction(
    *,
    horizon: int = 7,
    exog_data: t.Optional[pd.DataFrame] = None,
) -> dict:
    """Make a prediction using a saved SARIMAX model pipeline.

    Args:
        horizon: Number of days to forecast (default: 7)
        exog_data: Optional exogenous variables for the forecast period

    Returns:
        dict: Contains predictions, version, and metadata
    """

    results = {"predictions": None, "version": _version, "horizon": horizon}

    try:
        # SARIMAX forecast
        forecast = _sarimax_model.forecast(steps=horizon, exog=exog_data)

        # Convert to list for JSON serialization
        predictions = (
            forecast.tolist() if hasattr(forecast, "tolist") else list(forecast)
        )

        results = {
            "predictions": predictions,
            "version": _version,
            "horizon": horizon,
            "model_type": "sarimax",
        }

    except Exception as e:
        results["error"] = str(e)

    return results
