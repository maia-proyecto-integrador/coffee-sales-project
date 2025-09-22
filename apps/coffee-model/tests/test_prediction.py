import math
import pandas as pd

from model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # Given
    expected_horizon = 7
    
    # Create sample exogenous data for 7 days
    exog_data = pd.DataFrame({
        'is_holiday': [0, 0, 0, 0, 0, 1, 0],
        'is_holiday_prev': [0, 0, 0, 0, 0, 0, 1],
        'is_holiday_next': [0, 0, 0, 0, 1, 0, 0],
        'wx_temperature_2m': [22, 24, 23, 21, 25, 26, 24],
        'wx_precipitation': [0, 2, 0, 5, 0, 0, 1],
        'wx_cloudcover': [30, 60, 40, 80, 20, 10, 35]
    })

    # When
    result = make_prediction(horizon=expected_horizon, exog_data=exog_data)

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, list)
    assert result.get("errors") is None
    assert len(predictions) == expected_horizon
    assert result.get("version") is not None
    assert result.get("model_type") == "sarimax"
    assert result.get("horizon") == expected_horizon
    
    # Check that predictions are numeric
    for pred in predictions:
        assert isinstance(pred, (int, float))
        assert not math.isnan(pred)

