"""Scraper for Anthropic's /v1/models API endpoint and pricing page."""

from typing import Any

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.stdlib.get_logger()


class AnthropicAPIError(Exception):
    """Raised when the Anthropic API returns an error."""

    def __init__(self, status_code: int, message: str) -> None:
        """Initialize with status code and message."""
        self.status_code = status_code
        super().__init__(f"Anthropic API error {status_code}: {message}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
    reraise=True,
)
async def fetch_anthropic_models(
    client: httpx.AsyncClient, api_key: str
) -> list[dict[str, Any]]:
    """Fetch the list of models from Anthropic's /v1/models endpoint.

    Paginates using after_id + has_more until all models are fetched.

    Args:
        client: Shared httpx async client.
        api_key: Anthropic API key.

    Returns:
        List of raw model dicts with id, display_name, created_at.

    Raises:
        AnthropicAPIError: On authentication or unexpected errors.
        httpx.TimeoutException: On timeout (retried).
        httpx.HTTPStatusError: On server errors (retried).
    """
    all_models: list[dict[str, Any]] = []
    after_id: str | None = None

    while True:
        params: dict[str, str] = {"limit": "100"}
        if after_id:
            params["after_id"] = after_id

        response = await client.get(
            "https://api.anthropic.com/v1/models",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            params=params,
        )

        if response.status_code == 401:
            raise AnthropicAPIError(401, "Invalid API key")

        response.raise_for_status()

        data = response.json()
        models: list[dict[str, Any]] = data.get("data", [])
        all_models.extend(models)

        if not data.get("has_more", False) or not models:
            break

        after_id = models[-1].get("id")
        if not after_id:
            break

    await logger.ainfo("anthropic_scraper_complete", models_found=len(all_models))
    return all_models


async def scrape_anthropic_pricing(
    client: httpx.AsyncClient,
) -> dict[str, dict[str, Any]]:
    """Scrape pricing data from Anthropic's pricing page.

    Args:
        client: Shared httpx async client.

    Returns:
        Dict mapping model_id to pricing/capability data.
    """
    try:
        response = await client.get("https://www.anthropic.com/pricing")

        if response.status_code != 200:
            await logger.awarning(
                "anthropic_pricing_page_error",
                status_code=response.status_code,
            )
            return {}

        return _parse_anthropic_pricing_html(response.text)
    except Exception:
        await logger.awarning("anthropic_pricing_scrape_failed", exc_info=True)
        return {}


def _parse_anthropic_pricing_html(html: str) -> dict[str, dict[str, Any]]:
    """Parse pricing data from Anthropic's pricing page HTML.

    Args:
        html: Raw HTML of the pricing page.

    Returns:
        Dict mapping model_id to extracted pricing/capability data.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    result: dict[str, dict[str, Any]] = {}

    # Look for pricing tables or structured data
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        for row in rows[1:]:  # Skip header
            cells = row.find_all(["td", "th"])
            if len(cells) >= 3:
                model_text = cells[0].get_text(strip=True).lower()
                # Try to map display name to model ID
                model_id = _normalize_anthropic_model_name(model_text)
                if model_id:
                    entry: dict[str, Any] = {}
                    # Try to extract pricing from remaining cells
                    for cell in cells[1:]:
                        text = cell.get_text(strip=True)
                        price = _extract_price(text)
                        if price is not None:
                            if "input" not in entry:
                                entry["input_price_per_1m"] = price
                            elif "output" not in entry:
                                entry["output_price_per_1m"] = price
                    if entry:
                        result[model_id] = entry

    return result


def _normalize_anthropic_model_name(text: str) -> str:
    """Normalize a display name from the pricing page to a model ID.

    Args:
        text: Display name text from the page.

    Returns:
        Normalized model ID, or empty string if not recognized.
    """
    text = text.strip().lower()
    # Map common display names to model IDs
    name_map: dict[str, str] = {
        "claude opus 4": "claude-opus-4",
        "claude opus 4.6": "claude-opus-4-6",
        "claude sonnet 4.5": "claude-sonnet-4-5-20250929",
        "claude sonnet 4": "claude-sonnet-4-20250514",
        "claude haiku 4.5": "claude-haiku-4-5-20251001",
        "claude haiku 3.5": "claude-3-5-haiku-20241022",
        "claude 3.5 sonnet": "claude-3-5-sonnet-20241022",
        "claude 3 opus": "claude-3-opus-20240229",
    }
    for display_name, model_id in name_map.items():
        if display_name in text:
            return model_id

    # Try direct match for IDs already in text
    if text.startswith("claude-"):
        return text

    return ""


def _extract_price(text: str) -> float | None:
    """Extract a dollar price from text like '$3.00' or '$3 / MTok'.

    Args:
        text: Price text to parse.

    Returns:
        Price as float, or None if not parseable.
    """
    import re

    text = text.strip()
    if not text or text in {"—", "-", "n/a", "N/A"}:
        return None

    match = re.search(r"\$?([\d,]+\.?\d*)", text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            return None
    return None
