"""FastAPI application factory with lifespan management."""

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import httpx
import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from openai_models.config import Settings, get_settings
from openai_models.logging import setup_logging
from openai_models.routes import api_router
from openai_models.scraper.orchestrator import refresh_models
from openai_models.serialization import ORJSONResponse
from openai_models.store import ModelStore

logger = structlog.stdlib.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle: startup and shutdown."""
    settings: Settings = app.state.settings

    # 1. Setup logging
    setup_logging(settings.app_env, settings.log_level)

    # 2. Init store
    store = ModelStore(db_path=settings.db_file)
    await store.init_db()
    app.state.store = store

    # 3. Try loading from SQLite cache
    loaded = await store.load_from_db()
    if loaded:
        await logger.ainfo("models_loaded_from_cache")

    # 4. Create shared HTTP client
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        timeout=httpx.Timeout(float(settings.http_timeout)),
        http2=True,
        follow_redirects=True,
        headers={"User-Agent": "openai-model-api/1.0"},
    )
    app.state.http_client = http_client

    # 5. Record start time
    app.state.start_time = time.monotonic()

    # 6. Background refresh task
    async def _background_refresh() -> None:
        try:
            models = await refresh_models(http_client, settings)
            await store.replace_all(models)
            await logger.ainfo("initial_refresh_complete", models=len(models))
        except Exception:
            await logger.awarning("initial_refresh_failed", exc_info=True)

    refresh_task = asyncio.create_task(_background_refresh())

    # 7. Periodic refresh scheduler
    scheduler_task: asyncio.Task[None] | None = None
    if settings.refresh_interval_minutes > 0:

        async def _periodic_refresh() -> None:
            interval = settings.refresh_interval_minutes * 60
            while True:
                await asyncio.sleep(interval)
                try:
                    models = await refresh_models(http_client, settings)
                    await store.replace_all(models)
                    await logger.ainfo("periodic_refresh_complete", models=len(models))
                except Exception:
                    await logger.awarning("periodic_refresh_failed", exc_info=True)

        scheduler_task = asyncio.create_task(_periodic_refresh())

    await logger.ainfo("app_started", env=settings.app_env.value)

    yield

    # Shutdown
    refresh_task.cancel()
    if scheduler_task:
        scheduler_task.cancel()
    await http_client.aclose()
    await logger.ainfo("app_shutdown")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title="OpenAI Model Catalog API",
        description="Production-ready API for OpenAI model information",
        version="1.0.0",
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
    )

    app.state.settings = settings

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Lightweight timing middleware
    @app.middleware("http")
    async def add_response_time(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Add X-Response-Time header to every response."""
        start = time.monotonic()
        response = await call_next(request)
        elapsed_ms = round((time.monotonic() - start) * 1000, 2)
        response.headers["X-Response-Time"] = f"{elapsed_ms}ms"
        return response

    # Include routes
    app.include_router(api_router)

    return app
