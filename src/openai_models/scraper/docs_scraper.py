"""Scraper for OpenAI's documentation pages.

Targets:
- https://developers.openai.com/api/docs/models (model list)
- https://developers.openai.com/api/docs/models/compare (model comparison)
"""

import json
from typing import Any

import httpx
import structlog
from bs4 import BeautifulSoup

logger = structlog.stdlib.get_logger()

MODELS_PAGE_URL = "https://developers.openai.com/api/docs/models"
COMPARE_PAGE_URL = "https://developers.openai.com/api/docs/models/compare"


async def scrape_models_page(
    client: httpx.AsyncClient,
) -> dict[str, dict[str, Any]]:
    """Scrape model data from OpenAI's docs pages.

    First tries to extract structured JSON from Next.js hydration data.
    Falls back to HTML parsing with BeautifulSoup.

    Returns:
        Dict mapping model_id to enrichment data.
    """
    result: dict[str, dict[str, Any]] = {}

    # Scrape compare page (richest source)
    try:
        compare_data = await _scrape_compare_page(client)
        result.update(compare_data)
    except Exception:
        await logger.awarning("docs_scraper_compare_failed", exc_info=True)

    # Scrape models list page
    try:
        models_data = await _scrape_models_list_page(client)
        # Only add data for models not already enriched by compare page
        for model_id, data in models_data.items():
            if model_id not in result:
                result[model_id] = data
            else:
                # Merge missing fields
                for key, value in data.items():
                    if key not in result[model_id]:
                        result[model_id][key] = value
    except Exception:
        await logger.awarning("docs_scraper_models_page_failed", exc_info=True)

    await logger.ainfo("docs_scraper_complete", models_enriched=len(result))
    return result


async def _scrape_compare_page(
    client: httpx.AsyncClient,
) -> dict[str, dict[str, Any]]:
    """Scrape the model comparison page for detailed model data."""
    response = await client.get(COMPARE_PAGE_URL)
    response.raise_for_status()
    html = response.text

    # Try Next.js JSON extraction first
    result = _extract_nextjs_data(html)
    if result:
        return result

    # Fall back to HTML parsing
    return _parse_compare_html(html)


async def _scrape_models_list_page(
    client: httpx.AsyncClient,
) -> dict[str, dict[str, Any]]:
    """Scrape the models list page for names and descriptions."""
    response = await client.get(MODELS_PAGE_URL)
    response.raise_for_status()
    html = response.text

    # Try Next.js JSON extraction
    result = _extract_nextjs_data(html)
    if result:
        return result

    return _parse_models_list_html(html)


def _extract_nextjs_data(html: str) -> dict[str, dict[str, Any]]:
    """Try to extract model data from Next.js hydration scripts."""
    soup = BeautifulSoup(html, "lxml")
    result: dict[str, dict[str, Any]] = {}

    # Check for __NEXT_DATA__ script tag
    next_data_tag = soup.find("script", id="__NEXT_DATA__")
    tag_text = (
        next_data_tag.string
        if next_data_tag and hasattr(next_data_tag, "string")
        else None
    )
    if tag_text:
        try:
            data = json.loads(tag_text)
            models = _walk_for_models(data)
            if models:
                return models
        except (json.JSONDecodeError, KeyError):
            pass

    # Check for Next.js hydration push data
    for script in soup.find_all("script"):
        text = script.string or ""
        if "self.__next_f.push" in text:
            try:
                # Extract JSON from push calls
                for line in text.split("\n"):
                    if "self.__next_f.push" not in line:
                        continue
                    json_str = line.split("self.__next_f.push(", 1)[-1].rstrip(")")
                    try:
                        chunk = json.loads(json_str)
                        if isinstance(chunk, list) and len(chunk) > 1:
                            inner = chunk[1]
                            if isinstance(inner, str):
                                # Try parsing inner string as JSON
                                try:
                                    inner_data = json.loads(inner)
                                    models = _walk_for_models(inner_data)
                                    result.update(models)
                                except json.JSONDecodeError:
                                    pass
                    except json.JSONDecodeError:
                        pass
            except Exception:  # noqa: S110
                pass  # Best-effort extraction from hydration data

    return result


def _walk_for_models(data: Any) -> dict[str, dict[str, Any]]:
    """Walk a JSON structure looking for model definitions."""
    result: dict[str, dict[str, Any]] = {}

    if isinstance(data, dict):
        # Check if this dict looks like a model
        if "id" in data and ("context_window" in data or "pricing" in data):
            model_id = str(data["id"])
            result[model_id] = data
        else:
            for value in data.values():
                result.update(_walk_for_models(value))
    elif isinstance(data, list):
        for item in data:
            result.update(_walk_for_models(item))

    return result


def _parse_compare_html(html: str) -> dict[str, dict[str, Any]]:
    """Parse model data from comparison page HTML tables."""
    soup = BeautifulSoup(html, "lxml")
    result: dict[str, dict[str, Any]] = {}

    # Look for tables with model data
    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        if not headers:
            continue

        for row in table.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cells) < 2:
                continue

            row_data: dict[str, Any] = dict(zip(headers, cells, strict=False))
            model_id = row_data.get("model", row_data.get("id", ""))
            if model_id:
                result[model_id] = row_data

    return result


def _parse_models_list_html(html: str) -> dict[str, dict[str, Any]]:
    """Parse model names and descriptions from the models list page."""
    soup = BeautifulSoup(html, "lxml")
    result: dict[str, dict[str, Any]] = {}

    # Look for headings that might be model names
    for heading in soup.find_all(["h2", "h3", "h4"]):
        text = heading.get_text(strip=True)
        if not text:
            continue

        # Try to find a description in the next sibling
        desc = ""
        sibling = heading.find_next_sibling("p")
        if sibling:
            desc = sibling.get_text(strip=True)

        # Simple heuristic: if heading looks like a model name
        lower = text.lower()
        if any(prefix in lower for prefix in ("gpt-", "o1", "o3", "o4", "dall-e")):
            model_id = lower.replace(" ", "-")
            result[model_id] = {"name": text, "description": desc}

    return result
