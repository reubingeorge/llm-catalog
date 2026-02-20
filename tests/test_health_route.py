"""Tests for the health endpoint."""

import httpx


class TestHealthRoute:
    """Tests for GET /health."""

    async def test_health_response_shape(self, client: httpx.AsyncClient) -> None:
        """Health endpoint returns expected fields."""
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "models_loaded" in data
        assert "last_refreshed" in data
        assert "uptime_seconds" in data
        assert data["version"] == "1.0.0"

    async def test_health_model_count(self, client: httpx.AsyncClient) -> None:
        """Health reports correct model count from store."""
        resp = await client.get("/health")
        data = resp.json()
        # Our test store has 14 models (10 OpenAI + 2 Anthropic + 2 Gemini)
        assert data["models_loaded"] == 14

    async def test_health_uptime(self, client: httpx.AsyncClient) -> None:
        """Uptime is a positive number."""
        resp = await client.get("/health")
        data = resp.json()
        assert data["uptime_seconds"] >= 0
