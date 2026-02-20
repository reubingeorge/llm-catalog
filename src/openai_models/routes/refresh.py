"""Refresh endpoint to trigger model data re-scraping."""

import time
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException

from openai_models.dependencies import HttpClientDep, SettingsDep, StoreDep
from openai_models.models import RefreshRequest, RefreshResponse
from openai_models.scraper.orchestrator import refresh_models

router = APIRouter()


@router.post("/refresh", response_model=RefreshResponse)
async def refresh(
    store: StoreDep,
    client: HttpClientDep,
    settings: SettingsDep,
    body: RefreshRequest | None = None,
) -> RefreshResponse:
    """Trigger a manual refresh of model data.

    Returns 409 if a refresh is already in progress.
    """
    if body is None:
        body = RefreshRequest()

    # Try non-blocking acquire
    if store.refresh_lock.locked():
        raise HTTPException(status_code=409, detail="Refresh already in progress")

    start = time.monotonic()
    async with store.refresh_lock:
        models = await refresh_models(
            client,
            settings,
            probe_capabilities=body.probe_capabilities,
        )
        await store.replace_all_unlocked(models)

    duration = round(time.monotonic() - start, 2)
    now = datetime.now(tz=UTC)

    return RefreshResponse(
        status="completed",
        models_found=len(models),
        duration_seconds=duration,
        refreshed_at=now,
    )
