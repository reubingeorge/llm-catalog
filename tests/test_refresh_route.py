"""Tests for the refresh endpoint."""

import httpx
import respx

from openai_models.store import ModelStore


class TestRefreshRoute:
    """Tests for POST /refresh."""

    @respx.mock
    async def test_refresh_success(self, client: httpx.AsyncClient) -> None:
        """Successful refresh returns model count."""
        # Mock all outbound HTTP calls
        respx.get("https://api.openai.com/v1/models").mock(
            return_value=httpx.Response(200, json={"data": []})
        )
        respx.get("https://developers.openai.com/api/docs/models/compare").mock(
            return_value=httpx.Response(200, text="<html></html>")
        )
        respx.get("https://developers.openai.com/api/docs/models").mock(
            return_value=httpx.Response(200, text="<html></html>")
        )
        respx.get("https://platform.openai.com/docs/pricing").mock(
            return_value=httpx.Response(200, text="<html></html>")
        )

        resp = await client.post("/refresh", json={"probe_capabilities": False})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert "models_found" in data
        assert "duration_seconds" in data

    @respx.mock
    async def test_concurrent_refresh_409(
        self, client: httpx.AsyncClient, store: ModelStore
    ) -> None:
        """Concurrent refresh returns 409 when lock is held."""
        # Acquire the lock to simulate in-progress refresh
        async with store.refresh_lock:
            resp = await client.post("/refresh")
            assert resp.status_code == 409
