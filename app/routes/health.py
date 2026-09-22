from fastapi import APIRouter

from app import model_state
from app.schemas.prediction import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness/readiness probe",
    description="Deliberately carries no filesystem paths. Wired as the container health check.",
)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=True, model_type=model_state.get_model_name())
