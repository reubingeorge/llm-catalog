"""Orchestrator that coordinates all scrapers and merges data.

Merge priority per provider:
API response > Scraped pricing/docs > Hardcoded capability_map
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
from openai_models.scraper.anthropic_scraper import (
    fetch_anthropic_models,
    scrape_anthropic_pricing,
)
from openai_models.scraper.api_scraper import fetch_model_list
from openai_models.scraper.capability_map import (
    KNOWN_MODELS,
    get_known_model,
    infer_family,
)
from openai_models.scraper.docs_scraper import scrape_models_page
from openai_models.scraper.gemini_scraper import (
    fetch_gemini_models,
    scrape_gemini_pricing,
)
from openai_models.scraper.pricing_scraper import scrape_pricing

logger = structlog.stdlib.get_logger()


async def refresh_models(
    client: httpx.AsyncClient,
    settings: Settings,
    *,
    probe_capabilities: bool = False,
) -> list[OpenAIModel]:
    """Coordinate all scrapers and merge model data.

    Runs OpenAI, Anthropic, and Gemini providers concurrently.

    Args:
        client: Shared httpx async client.
        settings: Application settings.
        probe_capabilities: Whether to probe individual model capabilities.

    Returns:
        Merged list of OpenAIModel objects from all providers.
    """
    start = time.monotonic()
    semaphore = asyncio.Semaphore(settings.scrape_concurrency)
    now = datetime.now(tz=UTC)

    # Run all providers concurrently
    openai_task = asyncio.create_task(_refresh_openai(client, settings, semaphore, now))
    anthropic_task = asyncio.create_task(
        _refresh_anthropic(client, settings, semaphore, now)
    )
    gemini_task = asyncio.create_task(_refresh_gemini(client, settings, semaphore, now))

    openai_models = await openai_task
    anthropic_models = await anthropic_task
    gemini_models = await gemini_task

    all_models = openai_models + anthropic_models + gemini_models

    duration = time.monotonic() - start
    await logger.ainfo(
        "refresh_complete",
        models_found=len(all_models),
        openai_count=len(openai_models),
        anthropic_count=len(anthropic_models),
        gemini_count=len(gemini_models),
        duration_seconds=round(duration, 2),
    )

    return all_models


async def _refresh_openai(
    client: httpx.AsyncClient,
    settings: Settings,
    semaphore: asyncio.Semaphore,
    now: datetime,
) -> list[OpenAIModel]:
    """Refresh OpenAI models from API + docs + pricing + fallback."""
    api_models: list[dict[str, Any]] = []
    if settings.openai_api_key:
        try:
            async with semaphore:
                api_models = await fetch_model_list(client, settings.openai_api_key)
        except Exception:
            await logger.awarning("openai_api_scraper_failed", exc_info=True)

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
        await logger.awarning("openai_docs_scraper_failed", exc_info=True)

    try:
        pricing_data = await pricing_task
    except Exception:
        await logger.awarning("openai_pricing_scraper_failed", exc_info=True)

    # Build model set from API results
    model_ids: set[str] = set()
    api_model_data: dict[str, dict[str, Any]] = {}
    for m in api_models:
        mid = m.get("id", "")
        if mid:
            model_ids.add(mid)
            api_model_data[mid] = m

    # Fall back to known OpenAI models if API returned nothing
    if not model_ids:
        await logger.awarning("no_openai_api_models_falling_back_to_capability_map")
        model_ids = {
            k
            for k, v in KNOWN_MODELS.items()
            if v.get("provider", "openai") == "openai"
        }

    models: list[OpenAIModel] = []
    for model_id in model_ids:
        model = _merge_model(
            model_id=model_id,
            api_data=api_model_data.get(model_id, {}),
            docs_data=docs_data.get(model_id, {}),
            pricing_data=pricing_data.get(model_id),
            provider="openai",
            now=now,
        )
        models.append(model)

    return models


async def _refresh_anthropic(
    client: httpx.AsyncClient,
    settings: Settings,
    semaphore: asyncio.Semaphore,
    now: datetime,
) -> list[OpenAIModel]:
    """Refresh Anthropic models from API + pricing page + fallback."""
    api_models: list[dict[str, Any]] = []
    if settings.anthropic_api_key:
        try:
            async with semaphore:
                api_models = await fetch_anthropic_models(
                    client, settings.anthropic_api_key
                )
        except Exception:
            await logger.awarning("anthropic_api_scraper_failed", exc_info=True)

    scraped_pricing: dict[str, dict[str, Any]] = {}
    try:
        async with semaphore:
            scraped_pricing = await scrape_anthropic_pricing(client)
    except Exception:
        await logger.awarning("anthropic_pricing_scraper_failed", exc_info=True)

    # Build model set from API
    model_ids: set[str] = set()
    api_model_data: dict[str, dict[str, Any]] = {}
    for m in api_models:
        mid = m.get("id", "")
        if mid:
            model_ids.add(mid)
            api_model_data[mid] = m

    # Fall back to known Anthropic models
    if not model_ids:
        await logger.awarning("no_anthropic_api_models_falling_back_to_capability_map")
        model_ids = {
            k for k, v in KNOWN_MODELS.items() if v.get("provider") == "anthropic"
        }

    models: list[OpenAIModel] = []
    for model_id in model_ids:
        model = _merge_anthropic_model(
            model_id=model_id,
            api_data=api_model_data.get(model_id, {}),
            scraped_data=scraped_pricing.get(model_id, {}),
            now=now,
        )
        models.append(model)

    return models


async def _refresh_gemini(
    client: httpx.AsyncClient,
    settings: Settings,
    semaphore: asyncio.Semaphore,
    now: datetime,
) -> list[OpenAIModel]:
    """Refresh Gemini models from API + pricing page + fallback."""
    api_models: list[dict[str, Any]] = []
    if settings.gemini_api_key:
        try:
            async with semaphore:
                api_models = await fetch_gemini_models(client, settings.gemini_api_key)
        except Exception:
            await logger.awarning("gemini_api_scraper_failed", exc_info=True)

    scraped_pricing: dict[str, dict[str, Any]] = {}
    try:
        async with semaphore:
            scraped_pricing = await scrape_gemini_pricing(client)
    except Exception:
        await logger.awarning("gemini_pricing_scraper_failed", exc_info=True)

    # Build model set from API
    model_ids: set[str] = set()
    api_model_data: dict[str, dict[str, Any]] = {}
    for m in api_models:
        # Gemini API returns "models/gemini-2.5-pro" — strip prefix
        raw_name = m.get("name", "")
        mid = raw_name.removeprefix("models/") if raw_name else ""
        if mid:
            model_ids.add(mid)
            api_model_data[mid] = m

    # Fall back to known Gemini models
    if not model_ids:
        await logger.awarning("no_gemini_api_models_falling_back_to_capability_map")
        model_ids = {
            k for k, v in KNOWN_MODELS.items() if v.get("provider") == "google"
        }

    models: list[OpenAIModel] = []
    for model_id in model_ids:
        model = _merge_gemini_model(
            model_id=model_id,
            api_data=api_model_data.get(model_id, {}),
            scraped_data=scraped_pricing.get(model_id, {}),
            now=now,
        )
        models.append(model)

    return models


def _merge_model(
    model_id: str,
    api_data: dict[str, Any],
    docs_data: dict[str, Any],
    pricing_data: ModelPricing | None,
    provider: str,
    now: datetime,
) -> OpenAIModel:
    """Merge data from multiple sources for a single OpenAI model.

    Priority: API > docs > pricing page > hardcoded capability_map.
    """
    known = get_known_model(model_id)
    family = infer_family(model_id)

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
        provider=provider,
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


def _merge_anthropic_model(
    model_id: str,
    api_data: dict[str, Any],
    scraped_data: dict[str, Any],
    now: datetime,
) -> OpenAIModel:
    """Merge data for an Anthropic model. Priority: API > scraped > fallback."""
    known = get_known_model(model_id)
    family = infer_family(model_id)

    name = model_id
    description = ""
    context_window: int | None = None
    max_output_tokens: int | None = None
    capabilities = ModelCapabilities()
    pricing = ModelPricing()
    created_at: datetime | None = None

    # Fallback
    if known:
        name = known.name or model_id
        family = known.family or family
        description = known.description
        context_window = known.context_window
        max_output_tokens = known.max_output_tokens
        capabilities = known.capabilities
        pricing = known.pricing

    # Scraped pricing
    if scraped_data:
        inp = scraped_data.get("input_price_per_1m")
        out = scraped_data.get("output_price_per_1m")
        if inp is not None or out is not None:
            pricing = ModelPricing(
                input_price_per_1m=inp or pricing.input_price_per_1m,
                output_price_per_1m=out or pricing.output_price_per_1m,
                cached_input_price_per_1m=pricing.cached_input_price_per_1m,
            )

    # API data (highest priority)
    if api_data:
        display_name = api_data.get("display_name")
        if display_name:
            name = str(display_name)
        created_str = api_data.get("created_at")
        if created_str:
            try:
                created_at = datetime.fromisoformat(
                    str(created_str).replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

    return OpenAIModel(
        id=model_id,
        name=name,
        family=family,
        provider="anthropic",
        description=description,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        capabilities=capabilities,
        pricing=pricing,
        created_at=created_at,
        scraped_at=now,
    )


def _merge_gemini_model(
    model_id: str,
    api_data: dict[str, Any],
    scraped_data: dict[str, Any],
    now: datetime,
) -> OpenAIModel:
    """Merge data for a Gemini model. Priority: API > scraped > fallback."""
    known = get_known_model(model_id)
    family = infer_family(model_id)

    name = model_id
    description = ""
    context_window: int | None = None
    max_output_tokens: int | None = None
    capabilities = ModelCapabilities()
    pricing = ModelPricing()

    # Fallback
    if known:
        name = known.name or model_id
        family = known.family or family
        description = known.description
        context_window = known.context_window
        max_output_tokens = known.max_output_tokens
        capabilities = known.capabilities
        pricing = known.pricing

    # Scraped pricing
    if scraped_data:
        inp = scraped_data.get("input_price_per_1m")
        out = scraped_data.get("output_price_per_1m")
        if inp is not None or out is not None:
            pricing = ModelPricing(
                input_price_per_1m=inp or pricing.input_price_per_1m,
                output_price_per_1m=out or pricing.output_price_per_1m,
                cached_input_price_per_1m=pricing.cached_input_price_per_1m,
            )

    # API data (highest priority — Gemini API includes limits directly)
    if api_data:
        display_name = api_data.get("displayName")
        if display_name:
            name = str(display_name)
        desc = api_data.get("description")
        if desc:
            description = str(desc)
        input_limit = api_data.get("inputTokenLimit")
        if input_limit is not None:
            try:
                context_window = int(input_limit)
            except (ValueError, TypeError):
                pass
        output_limit = api_data.get("outputTokenLimit")
        if output_limit is not None:
            try:
                max_output_tokens = int(output_limit)
            except (ValueError, TypeError):
                pass

    return OpenAIModel(
        id=model_id,
        name=name,
        family=family,
        provider="google",
        description=description,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        capabilities=capabilities,
        pricing=pricing,
        scraped_at=now,
    )
