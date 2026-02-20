"""Scraper for Google Gemini's models API endpoint and pricing page."""

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


class GeminiAPIError(Exception):
    """Raised when the Gemini API returns an error."""

    def __init__(self, status_code: int, message: str) -> None:
        """Initialize with status code and message."""
        self.status_code = status_code
        super().__init__(f"Gemini API error {status_code}: {message}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
    reraise=True,
)
async def fetch_gemini_models(
    client: httpx.AsyncClient, api_key: str
) -> list[dict[str, Any]]:
    """Fetch the list of models from Google's Gemini API.

    Paginates with pageToken until all models are fetched.
    The API returns context/output limits directly.

    Args:
        client: Shared httpx async client.
        api_key: Google Gemini API key.

    Returns:
        List of raw model dicts with name, displayName, inputTokenLimit, etc.

    Raises:
        GeminiAPIError: On authentication or unexpected errors.
        httpx.TimeoutException: On timeout (retried).
        httpx.HTTPStatusError: On server errors (retried).
    """
    all_models: list[dict[str, Any]] = []
    page_token: str | None = None

    while True:
        params: dict[str, str] = {"key": api_key}
        if page_token:
            params["pageToken"] = page_token

        response = await client.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params=params,
        )

        if response.status_code == 401:
            raise GeminiAPIError(401, "Invalid API key")

        response.raise_for_status()

        data = response.json()
        models: list[dict[str, Any]] = data.get("models", [])
        all_models.extend(models)

        page_token = data.get("nextPageToken")
        if not page_token or not models:
            break

    await logger.ainfo("gemini_scraper_complete", models_found=len(all_models))
    return all_models


async def scrape_gemini_pricing(
    client: httpx.AsyncClient,
) -> dict[str, dict[str, Any]]:
    """Scrape pricing data from Google's Gemini pricing page.

    Args:
        client: Shared httpx async client.

    Returns:
        Dict mapping model_id to pricing/capability data.
    """
    try:
        response = await client.get("https://ai.google.dev/gemini-api/docs/pricing")

        if response.status_code != 200:
            await logger.awarning(
                "gemini_pricing_page_error",
                status_code=response.status_code,
            )
            return {}

        return _parse_gemini_pricing_html(response.text)
    except Exception:
        await logger.awarning("gemini_pricing_scrape_failed", exc_info=True)
        return {}


def _parse_gemini_pricing_html(html: str) -> dict[str, dict[str, Any]]:
    """Parse pricing data from Google's Gemini pricing page HTML.

    Args:
        html: Raw HTML of the pricing page.

    Returns:
        Dict mapping model_id to extracted pricing/capability data.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    result: dict[str, dict[str, Any]] = {}

    # Look for pricing tables
    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        for row in rows[1:]:  # Skip header
            cells = row.find_all(["td", "th"])
            if len(cells) >= 3:
                model_text = cells[0].get_text(strip=True).lower()
                model_id = _normalize_gemini_model_name(model_text)
                if model_id:
                    entry: dict[str, Any] = {}
                    for cell in cells[1:]:
                        text = cell.get_text(strip=True)
                        price = _extract_price(text)
                        if price is not None:
                            if "input_price_per_1m" not in entry:
                                entry["input_price_per_1m"] = price
                            elif "output_price_per_1m" not in entry:
                                entry["output_price_per_1m"] = price
                    if entry:
                        result[model_id] = entry

    return result


def _normalize_gemini_model_name(text: str) -> str:
    """Normalize a display name from the pricing page to a model ID.

    Args:
        text: Display name text from the page.

    Returns:
        Normalized model ID, or empty string if not recognized.
    """
    text = text.strip().lower()
    name_map: dict[str, str] = {
        "gemini 2.5 pro": "gemini-2.5-pro",
        "gemini 2.5 flash": "gemini-2.5-flash",
        "gemini 2.5 flash-lite": "gemini-2.5-flash-lite",
        "gemini 2.0 flash": "gemini-2.0-flash",
        "gemini 2.0 flash-lite": "gemini-2.0-flash-lite",
        "gemini 1.5 pro": "gemini-1.5-pro",
        "gemini 1.5 flash": "gemini-1.5-flash",
    }
    for display_name, model_id in name_map.items():
        if display_name in text:
            return model_id

    if text.startswith("gemini-"):
        return text

    return ""


def _extract_price(text: str) -> float | None:
    """Extract a dollar price from text.

    Args:
        text: Price text to parse.

    Returns:
        Price as float, or None if not parseable.
    """
    import re

    text = text.strip()
    if not text or text.lower() in {"—", "-", "n/a", "free"}:
        return None

    match = re.search(r"\$?([\d,]+\.?\d*)", text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            return None
    return None
