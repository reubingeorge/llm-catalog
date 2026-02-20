"""FastAPI dependency injection providers."""

from typing import Annotated

import httpx
from fastapi import Depends, Request

from openai_models.config import Settings
from openai_models.store import ModelStore


def get_store(request: Request) -> ModelStore:
    """Get the model store from app state."""
    store: ModelStore = request.app.state.store
    return store


def get_http_client(request: Request) -> httpx.AsyncClient:
    """Get the shared HTTP client from app state."""
    client: httpx.AsyncClient = request.app.state.http_client
    return client


def get_settings(request: Request) -> Settings:
    """Get application settings from app state."""
    settings: Settings = request.app.state.settings
    return settings


StoreDep = Annotated[ModelStore, Depends(get_store)]
HttpClientDep = Annotated[httpx.AsyncClient, Depends(get_http_client)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
