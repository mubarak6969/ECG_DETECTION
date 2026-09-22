from fastapi import APIRouter

from app import model_state
from app.schemas.prediction import ModelInfo

router = APIRouter()


@router.get(
    "/model-info",
    response_model=ModelInfo,
    summary="Model architecture and last-evaluated test metrics",
    description=(
        "Architecture, window/sampling-rate configuration, and the last-evaluated "
        "record-level test metrics (scripts/evaluate_model.py) - the same numbers "
        "shown as pills in the frontend UI."
    ),
)
def model_info() -> ModelInfo:
    return ModelInfo(**model_state.build_model_info())
