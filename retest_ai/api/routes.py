from fastapi import APIRouter, HTTPException, status
from typing import List, Union
from .schemas import (
    PreRetestEvent,
    BatchEventItem,
    BatchPredictionRequest,
    SinglePredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse
)
from ..models.service import MLService

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def get_health():
    """Service health check endpoint."""
    return {
        "status": "ok",
        "service": "ATE Retest-Benefit Prediction AI"
    }

@router.get("/model/info", response_model=ModelInfoResponse)
def get_model_info():
    """Retrieve active model information, feature whitelist, and training metadata."""
    ml_service = MLService.get_instance()
    return ml_service.get_model_info()

@router.post("/predict", response_model=SinglePredictionResponse)
def predict_single_event(event: PreRetestEvent):
    """
    Generate P(RETEST_BENEFICIAL) and a DOCX-reference recommendation
    for a single pre-retest failure event.
    Accepts ONLY approved pre-retest features.
    """
    ml_service = MLService.get_instance()
    try:
        result = ml_service.predict_single_event(event.model_dump())
        return result
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )

@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch_events(payload: Union[BatchPredictionRequest, List[BatchEventItem]]):
    """
    Generate P(RETEST_BENEFICIAL) and DOCX-reference recommendations
    for multiple pre-retest events.
    """
    ml_service = MLService.get_instance()
    if isinstance(payload, BatchPredictionRequest):
        event_list = [item.model_dump() for item in payload.events]
    else:
        event_list = [item.model_dump() for item in payload]

    try:
        predictions = ml_service.predict_batch_events(event_list)
        return {"predictions": predictions}
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {str(e)}"
        )
