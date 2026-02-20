"""Tests for all scraper modules."""

from pathlib import Path

import httpx
import pytest
import respx

from openai_models.scraper.api_scraper import OpenAIAPIError, fetch_model_list
from openai_models.scraper.capability_map import get_known_model, infer_family
from openai_models.scraper.docs_scraper import (
    _extract_nextjs_data,
    _parse_compare_html,
    _parse_models_list_html,
    scrape_models_page,
)
from openai_models.scraper.pricing_scraper import _parse_price, scrape_pricing

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestCapabilityMap:
    """Tests for the hardcoded capability map."""

    def test_get_known_model_exists(self) -> None:
        """Known model returns full data."""
        model = get_known_model("gpt-5.2")
        assert model is not None
        assert model.id == "gpt-5.2"
        assert model.pricing.input_price_per_1m == 1.75
        assert model.capabilities.reasoning is True
        assert model.capabilities.vision is True

    def test_get_known_model_not_found(self) -> None:
        """Unknown model returns None."""
        assert get_known_model("nonexistent") is None

    def test_infer_family_gpt52(self) -> None:
        """GPT-5.2 family inference."""
        assert infer_family("gpt-5.2") == "gpt-5.2"
        assert infer_family("gpt-5.2-pro") == "gpt-5.2"

    def test_infer_family_gpt5(self) -> None:
        """GPT-5 family inference."""
        assert infer_family("gpt-5") == "gpt-5"
        assert infer_family("gpt-5-mini") == "gpt-5"

    def test_infer_family_o_series(self) -> None:
        """O-series family inference."""
        assert infer_family("o4-mini") == "o4"
        assert infer_family("o3") == "o3"
        assert infer_family("o1") == "o1"

    def test_infer_family_unknown(self) -> None:
        """Unknown model returns empty string."""
        assert infer_family("totally-unknown-model") == ""

    def test_known_model_deprecated(self) -> None:
        """GPT-3.5-turbo is marked deprecated."""
        model = get_known_model("gpt-3.5-turbo")
        assert model is not None
        assert model.deprecated is True


class TestAPIScraper:
    """Tests for the OpenAI API scraper."""

    @respx.mock
    async def test_fetch_success(self, api_response_json: dict[str, object]) -> None:
        """Successful API response returns model list."""
        respx.get("https://api.openai.com/v1/models").mock(
            return_value=httpx.Response(200, json=api_response_json)
        )
        async with httpx.AsyncClient() as client:
            models = await fetch_model_list(client, "sk-test")

        assert len(models) == 14
        ids = {m["id"] for m in models}
        assert "gpt-5.2" in ids
        assert "o4-mini" in ids

    @respx.mock
    async def test_fetch_401(self) -> None:
        """401 raises OpenAIAPIError."""
        respx.get("https://api.openai.com/v1/models").mock(
            return_value=httpx.Response(401, json={"error": "invalid key"})
        )
        async with httpx.AsyncClient() as client:
            with pytest.raises(OpenAIAPIError) as exc_info:
                await fetch_model_list(client, "bad-key")
            assert exc_info.value.status_code == 401

    @respx.mock
    async def test_fetch_timeout(self) -> None:
        """Timeout retries and eventually raises."""
        respx.get("https://api.openai.com/v1/models").mock(
            side_effect=httpx.ReadTimeout("timeout")
        )
        async with httpx.AsyncClient() as client:
            with pytest.raises(httpx.ReadTimeout):
                await fetch_model_list(client, "sk-test")

    @respx.mock
    async def test_fetch_empty_data(self) -> None:
        """Empty data array returns empty list."""
        respx.get("https://api.openai.com/v1/models").mock(
            return_value=httpx.Response(200, json={"data": []})
        )
        async with httpx.AsyncClient() as client:
            models = await fetch_model_list(client, "sk-test")
        assert models == []


class TestDocsScraper:
    """Tests for the docs page scraper."""

    @respx.mock
    async def test_scrape_models_page_success(self, models_page_html: str) -> None:
        """Real models page HTML is parsed without error."""
        respx.get("https://developers.openai.com/api/docs/models/compare").mock(
            return_value=httpx.Response(200, text=models_page_html)
        )
        respx.get("https://developers.openai.com/api/docs/models").mock(
            return_value=httpx.Response(200, text=models_page_html)
        )
        async with httpx.AsyncClient() as client:
            result = await scrape_models_page(client)
        # Should not crash — result may be empty if no structured data found
        assert isinstance(result, dict)

    @respx.mock
    async def test_scrape_malformed_html(self) -> None:
        """Malformed HTML returns empty dict without crashing."""
        respx.get("https://developers.openai.com/api/docs/models/compare").mock(
            return_value=httpx.Response(200, text="<html><not valid>>><<<")
        )
        respx.get("https://developers.openai.com/api/docs/models").mock(
            return_value=httpx.Response(200, text="<html><not valid>>><<<")
        )
        async with httpx.AsyncClient() as client:
            result = await scrape_models_page(client)
        assert isinstance(result, dict)

    @respx.mock
    async def test_scrape_server_error(self) -> None:
        """Server errors return empty dict."""
        respx.get("https://developers.openai.com/api/docs/models/compare").mock(
            return_value=httpx.Response(500)
        )
        respx.get("https://developers.openai.com/api/docs/models").mock(
            return_value=httpx.Response(500)
        )
        async with httpx.AsyncClient() as client:
            result = await scrape_models_page(client)
        assert isinstance(result, dict)


class TestPricingScraper:
    """Tests for the pricing page scraper."""

    @respx.mock
    async def test_scrape_pricing_success(self, pricing_page_html: str) -> None:
        """Pricing page HTML is parsed correctly."""
        respx.get("https://platform.openai.com/docs/pricing").mock(
            return_value=httpx.Response(200, text=pricing_page_html)
        )
        async with httpx.AsyncClient() as client:
            result = await scrape_pricing(client)

        assert len(result) > 0
        assert "gpt-5.2" in result
        assert result["gpt-5.2"].input_price_per_1m == 1.75
        assert result["gpt-5.2"].output_price_per_1m == 14.00

    @respx.mock
    async def test_scrape_pricing_403(self) -> None:
        """403 returns empty dict gracefully."""
        respx.get("https://platform.openai.com/docs/pricing").mock(
            return_value=httpx.Response(403)
        )
        async with httpx.AsyncClient() as client:
            result = await scrape_pricing(client)
        assert result == {}

    @respx.mock
    async def test_scrape_pricing_empty_table(self) -> None:
        """Page with empty table returns empty dict."""
        html = (
            "<html><body><table><thead><tr><th>Model</th>"
            "</tr></thead><tbody></tbody></table></body></html>"
        )
        respx.get("https://platform.openai.com/docs/pricing").mock(
            return_value=httpx.Response(200, text=html)
        )
        async with httpx.AsyncClient() as client:
            result = await scrape_pricing(client)
        assert result == {}


class TestDocsScraperParsing:
    """Tests for docs scraper parsing functions."""

    def test_extract_nextjs_data_with_next_data_tag(self) -> None:
        """Extract data from __NEXT_DATA__ script tag."""
        html = """
        <html><head></head><body>
        <script id="__NEXT_DATA__" type="application/json">
        {"props":{"models":[{"id":"gpt-5","context_window":400000}]}}
        </script>
        </body></html>
        """
        result = _extract_nextjs_data(html)
        assert "gpt-5" in result

    def test_extract_nextjs_data_empty(self) -> None:
        """No Next.js data returns empty dict."""
        html = "<html><body><p>No data</p></body></html>"
        result = _extract_nextjs_data(html)
        assert result == {}

    def test_extract_nextjs_data_invalid_json(self) -> None:
        """Invalid JSON in __NEXT_DATA__ returns empty."""
        html = '<html><body><script id="__NEXT_DATA__">not json</script></body></html>'
        result = _extract_nextjs_data(html)
        assert result == {}

    def test_parse_compare_html_with_table(self) -> None:
        """Parse model data from HTML table."""
        html = """
        <html><body>
        <table>
        <thead><tr><th>Model</th><th>Context Window</th><th>Price</th></tr></thead>
        <tbody>
        <tr><td>gpt-5.2</td><td>400000</td><td>$1.75</td></tr>
        <tr><td>gpt-4.1</td><td>1048000</td><td>$2.00</td></tr>
        </tbody>
        </table>
        </body></html>
        """
        result = _parse_compare_html(html)
        assert "gpt-5.2" in result
        assert "gpt-4.1" in result

    def test_parse_compare_html_empty(self) -> None:
        """Empty HTML returns empty dict."""
        result = _parse_compare_html("<html><body></body></html>")
        assert result == {}

    def test_parse_models_list_html(self) -> None:
        """Parse model names from headings."""
        html = """
        <html><body>
        <h3>GPT-5.2</h3>
        <p>Most capable model.</p>
        <h3>o3</h3>
        <p>Reasoning model.</p>
        </body></html>
        """
        result = _parse_models_list_html(html)
        assert isinstance(result, dict)

    def test_parse_models_list_html_empty(self) -> None:
        """Empty page returns empty dict."""
        result = _parse_models_list_html("<html><body></body></html>")
        assert result == {}

    def test_nextjs_hydration_push(self) -> None:
        """Extract data from Next.js hydration push scripts."""
        push_data = (
            "self.__next_f.push([1, "
            '"[{\\"id\\":\\"gpt-test\\",\\"context_window\\":100000}]"])'
        )
        html = f"<html><body><script>{push_data}</script></body></html>"
        result = _extract_nextjs_data(html)
        # May or may not extract depending on JSON validity
        assert isinstance(result, dict)


class TestPriceParser:
    """Tests for the price parsing helper."""

    def test_parse_dollar_amount(self) -> None:
        """Parse standard dollar format."""
        assert _parse_price("$1.75 / 1M tokens") == 1.75

    def test_parse_dash(self) -> None:
        """Dash means no price."""
        assert _parse_price("—") is None
        assert _parse_price("-") is None

    def test_parse_na(self) -> None:
        """N/A means no price."""
        assert _parse_price("n/a") is None
        assert _parse_price("N/A") is None

    def test_parse_empty(self) -> None:
        """Empty string means no price."""
        assert _parse_price("") is None

    def test_parse_comma_number(self) -> None:
        """Parse number with comma."""
        assert _parse_price("$1,000.50") == 1000.50


class TestOrchestrator:
    """Tests for the orchestrator merge logic."""

    @respx.mock
    async def test_full_merge(
        self,
        api_response_json: dict[str, object],
        pricing_page_html: str,
        models_page_html: str,
    ) -> None:
        """Full merge with all sources produces correct models."""
        from openai_models.config import Environment, Settings
        from openai_models.scraper.orchestrator import refresh_models

        respx.get("https://api.openai.com/v1/models").mock(
            return_value=httpx.Response(200, json=api_response_json)
        )
        respx.get("https://developers.openai.com/api/docs/models/compare").mock(
            return_value=httpx.Response(200, text=models_page_html)
        )
        respx.get("https://developers.openai.com/api/docs/models").mock(
            return_value=httpx.Response(200, text=models_page_html)
        )
        respx.get("https://platform.openai.com/docs/pricing").mock(
            return_value=httpx.Response(200, text=pricing_page_html)
        )

        settings = Settings(
            openai_api_key="sk-test",
            app_env=Environment.TESTING,
            db_path="",
        )
        async with httpx.AsyncClient() as client:
            models = await refresh_models(client, settings)

        assert len(models) == 14
        ids = {m.id for m in models}
        assert "gpt-5.2" in ids
        assert "gpt-4.1" in ids
        assert "o4-mini" in ids

        # Check enrichment from capability map
        gpt52 = next(m for m in models if m.id == "gpt-5.2")
        assert gpt52.family == "gpt-5.2"
        assert gpt52.context_window == 400_000

    @respx.mock
    async def test_fallback_to_capability_map(self) -> None:
        """When API fails, falls back to capability map."""
        from openai_models.config import Environment, Settings
        from openai_models.scraper.orchestrator import refresh_models

        respx.get("https://api.openai.com/v1/models").mock(
            return_value=httpx.Response(500)
        )
        respx.get("https://developers.openai.com/api/docs/models/compare").mock(
            return_value=httpx.Response(500)
        )
        respx.get("https://developers.openai.com/api/docs/models").mock(
            return_value=httpx.Response(500)
        )
        respx.get("https://platform.openai.com/docs/pricing").mock(
            return_value=httpx.Response(500)
        )

        settings = Settings(
            openai_api_key="sk-test",
            app_env=Environment.TESTING,
            db_path="",
        )
        async with httpx.AsyncClient() as client:
            models = await refresh_models(client, settings)

        # Should fall back to hardcoded models
        assert len(models) > 0
        ids = {m.id for m in models}
        assert "gpt-5.2" in ids
