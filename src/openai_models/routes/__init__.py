"""API route handlers."""

from fastapi import APIRouter

from openai_models.routes.health import router as health_router
from openai_models.routes.models import router as models_router
from openai_models.routes.refresh import router as refresh_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(models_router)
api_router.include_router(refresh_router)

__all__ = ["api_router"]
