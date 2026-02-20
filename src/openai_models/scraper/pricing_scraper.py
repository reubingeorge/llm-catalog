"""Scraper for OpenAI's pricing page."""

import re
from typing import Any

import httpx
import structlog
from bs4 import BeautifulSoup

from openai_models.models import ModelPricing

logger = structlog.stdlib.get_logger()

PRICING_URL = "https://platform.openai.com/docs/pricing"


async def scrape_pricing(
    client: httpx.AsyncClient,
) -> dict[str, ModelPricing]:
    """Scrape pricing data from OpenAI's pricing page.

    Returns:
        Dict mapping model_id to ModelPricing. Empty dict on failure.
    """
    try:
        response = await client.get(PRICING_URL)

        if response.status_code == 403:
            await logger.awarning("pricing_page_returned_403")
            return {}

        response.raise_for_status()
        return _parse_pricing_html(response.text)
    except httpx.HTTPStatusError:
        await logger.awarning("pricing_scraper_http_error", exc_info=True)
        return {}
    except Exception:
        await logger.awarning("pricing_scraper_failed", exc_info=True)
        return {}


def _parse_pricing_html(html: str) -> dict[str, ModelPricing]:
    """Parse pricing tables from the HTML page."""
    soup = BeautifulSoup(html, "lxml")
    result: dict[str, ModelPricing] = {}

    for table in soup.find_all("table"):
        headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]
        if not headers:
            continue

        for row in table.find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cells) < 2:
                continue

            row_data: dict[str, Any] = dict(zip(headers, cells, strict=False))
            model_id = row_data.get("model", row_data.get("name", ""))
            if not model_id:
                continue

            model_id = model_id.lower().strip()
            input_price = _parse_price(row_data.get("input", ""))
            output_price = _parse_price(row_data.get("output", ""))
            cached_price = _parse_price(row_data.get("cached input", ""))

            if input_price is not None or output_price is not None:
                result[model_id] = ModelPricing(
                    input_price_per_1m=input_price,
                    output_price_per_1m=output_price,
                    cached_input_price_per_1m=cached_price,
                )

    await_log_msg = f"pricing_scraper_complete: {len(result)} models"
    logger.info(await_log_msg)
    return result


def _parse_price(text: str) -> float | None:
    """Extract a dollar amount from text like '$2.50 / 1M tokens'."""
    if not text or text.strip() in ("—", "-", "n/a", "N/A"):
        return None
    match = re.search(r"\$?([\d,.]+)", text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            return None
    return None
