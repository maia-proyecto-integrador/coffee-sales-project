from typing import Any, List, Optional

from pydantic import BaseModel, Field

# Esquema para variables exógenas individuales
class ExogDataPoint(BaseModel):
    is_holiday: int = Field(..., description="¿Es día festivo? (0=No, 1=Sí)")
    is_holiday_prev: int = Field(..., description="¿Ayer fue festivo? (0=No, 1=Sí)")
    is_holiday_next: int = Field(..., description="¿Mañana es festivo? (0=No, 1=Sí)")
    wx_temperature_2m: float = Field(..., description="Temperatura en grados Celsius")
    wx_precipitation: float = Field(..., description="Precipitación en mm")
    wx_cloudcover: float = Field(..., description="Cobertura de nubes en %")

# Esquema de entrada para predicciones de café
class ForecastInputs(BaseModel):
    horizon: int = Field(default=7, ge=1, le=30, description="Número de días a predecir (1-30)")
    exog_data: Optional[List[ExogDataPoint]] = Field(
        None, 
        description="Variables exógenas para cada día del horizonte de predicción"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "horizon": 7,
                "exog_data": [
                    {
                        "is_holiday": 0,
                        "is_holiday_prev": 0,
                        "is_holiday_next": 0,
                        "wx_temperature_2m": 22.5,
                        "wx_precipitation": 0.0,
                        "wx_cloudcover": 30.0
                    },
                    {
                        "is_holiday": 0,
                        "is_holiday_prev": 0,
                        "is_holiday_next": 0,
                        "wx_temperature_2m": 24.0,
                        "wx_precipitation": 2.0,
                        "wx_cloudcover": 60.0
                    },
                    {
                        "is_holiday": 0,
                        "is_holiday_prev": 0,
                        "is_holiday_next": 0,
                        "wx_temperature_2m": 23.0,
                        "wx_precipitation": 0.0,
                        "wx_cloudcover": 40.0
                    },
                    {
                        "is_holiday": 0,
                        "is_holiday_prev": 0,
                        "is_holiday_next": 1,
                        "wx_temperature_2m": 21.0,
                        "wx_precipitation": 5.0,
                        "wx_cloudcover": 80.0
                    },
                    {
                        "is_holiday": 1,
                        "is_holiday_prev": 0,
                        "is_holiday_next": 0,
                        "wx_temperature_2m": 25.0,
                        "wx_precipitation": 0.0,
                        "wx_cloudcover": 20.0
                    },
                    {
                        "is_holiday": 0,
                        "is_holiday_prev": 1,
                        "is_holiday_next": 0,
                        "wx_temperature_2m": 26.0,
                        "wx_precipitation": 0.0,
                        "wx_cloudcover": 10.0
                    },
                    {
                        "is_holiday": 0,
                        "is_holiday_prev": 0,
                        "is_holiday_next": 0,
                        "wx_temperature_2m": 24.0,
                        "wx_precipitation": 1.0,
                        "wx_cloudcover": 35.0
                    }
                ]
            }
        }

# Esquema de los resultados de predicción
class PredictionResults(BaseModel):
    predictions: Optional[List[float]] = Field(None, description="Lista de predicciones para cada día")
    version: str = Field(..., description="Versión del modelo")
    horizon: int = Field(..., description="Número de días predichos")
    model_type: Optional[str] = Field(None, description="Tipo de modelo utilizado")
    error: Optional[str] = Field(None, description="Mensaje de error si ocurrió algún problema")

# Esquema para inputs múltiples (mantenido para compatibilidad)
class MultipleDataInputs(BaseModel):
    inputs: List[dict]

    class Config:
        json_schema_extra = {
            "example": {
                "inputs": [
                    {
                        "Customer_Age": 57,
                        "Gender": "M",
                        "Dependent_count": 4,
                        "Education_Level": "Graduate",
                        "Marital_Status": "Single",
                        "Income_Category": "$120K +",
                        "Card_Category": "Blue",
                        "Months_on_book":52,
                        "Total_Relationship_Count":2,
                        "Months_Inactive_12_mon":3,
                        "Contacts_Count_12_mon":2,
                        "Credit_Limit":25808,
                        "Total_Revolving_Bal":0,
                        "Avg_Open_To_Buy":25808,
                        "Total_Amt_Chng_Q4_Q1":0.712,
                        "Total_Trans_Amt":7794,
                        "Total_Trans_Ct":94,
                        "Total_Ct_Chng_Q4_Q1":0.843,
                        "Avg_Utilization_Ratio": 0
                    }
                ]
            }
        }
