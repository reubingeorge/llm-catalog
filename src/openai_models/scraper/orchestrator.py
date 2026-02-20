"""Orchestrator that coordinates all scrapers and merges data.

Merge priority:
API /v1/models > Compare page > Models list > Pricing > capability_map
"""

import asyncio
import time
from datetime import UTC, datetime
from typing import Any

import httpx
import structlog

from openai_models.config import Settings
from openai_models.models import (
    ModelCapabilities,
    ModelPricing,
    OpenAIModel,
)
from openai_models.scraper.api_scraper import fetch_model_list
from openai_models.scraper.capability_map import (
    KNOWN_MODELS,
    get_known_model,
    infer_family,
)
from openai_models.scraper.docs_scraper import scrape_models_page
from openai_models.scraper.pricing_scraper import scrape_pricing

logger = structlog.stdlib.get_logger()


async def refresh_models(
    client: httpx.AsyncClient,
    settings: Settings,
    *,
    probe_capabilities: bool = False,
) -> list[OpenAIModel]:
    """Coordinate all scrapers and merge model data.

    Args:
        client: Shared httpx async client.
        settings: Application settings.
        probe_capabilities: Whether to probe individual model capabilities.

    Returns:
        Merged list of OpenAIModel objects.
    """
    start = time.monotonic()
    semaphore = asyncio.Semaphore(settings.scrape_concurrency)
    now = datetime.now(tz=UTC)

    # Step 1: Fetch model IDs from API
    api_models: list[dict[str, Any]] = []
    if settings.openai_api_key:
        try:
            async with semaphore:
                api_models = await fetch_model_list(client, settings.openai_api_key)
        except Exception:
            await logger.awarning("api_scraper_failed", exc_info=True)

    # Step 2: Fetch enrichment data concurrently
    docs_data: dict[str, dict[str, Any]] = {}
    pricing_data: dict[str, ModelPricing] = {}

    async def _fetch_docs() -> dict[str, dict[str, Any]]:
        async with semaphore:
            return await scrape_models_page(client)

    async def _fetch_pricing() -> dict[str, ModelPricing]:
        async with semaphore:
            return await scrape_pricing(client)

    docs_task = asyncio.create_task(_fetch_docs())
    pricing_task = asyncio.create_task(_fetch_pricing())

    try:
        docs_data = await docs_task
    except Exception:
        await logger.awarning("docs_scraper_failed_in_orchestrator", exc_info=True)

    try:
        pricing_data = await pricing_task
    except Exception:
        await logger.awarning("pricing_scraper_failed_in_orchestrator", exc_info=True)

    # Step 3: Build model set from API results
    model_ids: set[str] = set()
    api_model_data: dict[str, dict[str, Any]] = {}
    for m in api_models:
        mid = m.get("id", "")
        if mid:
            model_ids.add(mid)
            api_model_data[mid] = m

    # If API returned nothing, fall back to known models
    if not model_ids:
        await logger.awarning("no_api_models_falling_back_to_capability_map")
        model_ids = set(KNOWN_MODELS.keys())

    # Step 4: Merge data for each model
    models: list[OpenAIModel] = []
    for model_id in model_ids:
        model = _merge_model(
            model_id=model_id,
            api_data=api_model_data.get(model_id, {}),
            docs_data=docs_data.get(model_id, {}),
            pricing_data=pricing_data.get(model_id),
            now=now,
        )
        models.append(model)

    duration = time.monotonic() - start
    await logger.ainfo(
        "refresh_complete",
        models_found=len(models),
        duration_seconds=round(duration, 2),
        api_models=len(api_models),
        docs_enriched=len(docs_data),
        pricing_enriched=len(pricing_data),
    )

    return models


def _merge_model(
    model_id: str,
    api_data: dict[str, Any],
    docs_data: dict[str, Any],
    pricing_data: ModelPricing | None,
    now: datetime,
) -> OpenAIModel:
    """Merge data from multiple sources for a single model.

    Priority: API > docs > pricing page > hardcoded capability_map.
    """
    known = get_known_model(model_id)
    family = infer_family(model_id)

    # Start with fallback values
    name = model_id
    description = ""
    context_window: int | None = None
    max_output_tokens: int | None = None
    knowledge_cutoff: str | None = None
    deprecated = False
    capabilities = ModelCapabilities()
    pricing = ModelPricing()
    endpoints: list[str] = []
    created_at: datetime | None = None

    # Layer 5: Hardcoded capability map (lowest priority)
    if known:
        name = known.name or model_id
        family = known.family or family
        description = known.description
        context_window = known.context_window
        max_output_tokens = known.max_output_tokens
        deprecated = known.deprecated
        capabilities = known.capabilities
        pricing = known.pricing

    # Layer 4: Pricing page
    if pricing_data:
        inp = pricing_data.input_price_per_1m or pricing.input_price_per_1m
        out = pricing_data.output_price_per_1m or pricing.output_price_per_1m
        cached = (
            pricing_data.cached_input_price_per_1m or pricing.cached_input_price_per_1m
        )
        pricing = ModelPricing(
            input_price_per_1m=inp,
            output_price_per_1m=out,
            cached_input_price_per_1m=cached,
        )

    # Layer 3/2: Docs data (compare page + models list page)
    if docs_data:
        name = str(docs_data.get("name", name)) or name
        description = str(docs_data.get("description", description)) or description
        if "context_window" in docs_data:
            try:
                context_window = int(docs_data["context_window"])
            except (ValueError, TypeError):
                pass
        if "max_output_tokens" in docs_data:
            try:
                max_output_tokens = int(docs_data["max_output_tokens"])
            except (ValueError, TypeError):
                pass
        if "knowledge_cutoff" in docs_data:
            knowledge_cutoff = str(docs_data["knowledge_cutoff"])
        if "endpoints" in docs_data and isinstance(docs_data["endpoints"], list):
            endpoints = [str(e) for e in docs_data["endpoints"]]

    # Layer 1: API data (highest priority for existence + created_at)
    if api_data:
        if "created" in api_data:
            try:
                created_at = datetime.fromtimestamp(int(api_data["created"]), tz=UTC)
            except (ValueError, TypeError, OSError):
                pass

    return OpenAIModel(
        id=model_id,
        name=name,
        family=family,
        description=description,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        knowledge_cutoff=knowledge_cutoff,
        deprecated=deprecated,
        capabilities=capabilities,
        pricing=pricing,
        endpoints=endpoints,
        created_at=created_at,
        scraped_at=now,
    )
