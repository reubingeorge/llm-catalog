"""Scraper for OpenAI's /v1/models API endpoint."""

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


class OpenAIAPIError(Exception):
    """Raised when the OpenAI API returns an error."""

    def __init__(self, status_code: int, message: str) -> None:
        """Initialize with status code and message."""
        self.status_code = status_code
        super().__init__(f"OpenAI API error {status_code}: {message}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
    reraise=True,
)
async def fetch_model_list(
    client: httpx.AsyncClient, api_key: str
) -> list[dict[str, Any]]:
    """Fetch the list of models from OpenAI's /v1/models endpoint.

    Args:
        client: Shared httpx async client.
        api_key: OpenAI API key.

    Returns:
        List of raw model dicts with id, created, owned_by.

    Raises:
        OpenAIAPIError: On authentication or unexpected errors.
        httpx.TimeoutException: On timeout (retried).
        httpx.HTTPStatusError: On server errors (retried).
    """
    response = await client.get(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )

    if response.status_code == 401:
        raise OpenAIAPIError(401, "Invalid API key")

    response.raise_for_status()

    data = response.json()
    models: list[dict[str, Any]] = data.get("data", [])

    await logger.ainfo("api_scraper_complete", models_found=len(models))
    return models
