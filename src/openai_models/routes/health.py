"""Health check endpoint."""

import time

from fastapi import APIRouter, Request

from openai_models import __version__
from openai_models.dependencies import StoreDep
from openai_models.models import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request, store: StoreDep) -> HealthResponse:
    """Return API health status."""
    snapshot = store.get_snapshot()
    start_time: float = request.app.state.start_time
    return HealthResponse(
        status="healthy",
        models_loaded=len(snapshot.models),
        last_refreshed=snapshot.last_refreshed if snapshot.models else None,
        uptime_seconds=round(time.monotonic() - start_time, 1),
        version=__version__,
    )
