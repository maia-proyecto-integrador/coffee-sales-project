from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from loguru import logger
from model import __version__ as model_version
from model.predict import make_prediction

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()

# Ruta para verificar que la API se esté ejecutando correctamente
@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Health Check endpoint
    """
    
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()

# Ruta para realizar las predicciones
@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.ForecastInputs) -> Any:
    """
    Prediccion de ventas de café usando modelo SARIMAX
    """

    logger.info(f"Making prediction with horizon: {input_data.horizon}")
    
    # Convertir datos exógenos a DataFrame si se proporcionan
    exog_data = None
    if input_data.exog_data:
        exog_data = pd.DataFrame(jsonable_encoder(input_data.exog_data))
        logger.info(f"Exogenous data provided: {exog_data.shape}")

    # Hacer predicción
    results = make_prediction(horizon=input_data.horizon, exog_data=exog_data)

    # Verificar errores
    if "error" in results:
        logger.warning(f"Prediction error: {results.get('error')}")
        raise HTTPException(status_code=400, detail=results["error"])

    logger.info(f"Prediction results: {len(results.get('predictions', []))} predictions generated")

    return results
