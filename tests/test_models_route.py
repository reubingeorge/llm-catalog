"""Tests for the models route endpoints."""

import httpx


class TestListModels:
    """Tests for GET /models."""

    async def test_list_all(self, client: httpx.AsyncClient) -> None:
        """Returns all non-deprecated models by default."""
        resp = await client.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] > 0
        # Default excludes deprecated
        ids = {m["id"] for m in data["models"]}
        assert "gpt-3.5-turbo" not in ids

    async def test_include_deprecated(self, client: httpx.AsyncClient) -> None:
        """include_deprecated=true returns all models."""
        resp = await client.get("/models?include_deprecated=true")
        data = resp.json()
        ids = {m["id"] for m in data["models"]}
        assert "gpt-3.5-turbo" in ids

    async def test_filter_vision(self, client: httpx.AsyncClient) -> None:
        """vision=true returns only vision models."""
        resp = await client.get("/models?vision=true")
        data = resp.json()
        for m in data["models"]:
            assert m["capabilities"]["vision"] is True

    async def test_filter_reasoning(self, client: httpx.AsyncClient) -> None:
        """reasoning=true returns only reasoning models."""
        resp = await client.get("/models?reasoning=true")
        data = resp.json()
        assert data["count"] > 0
        for m in data["models"]:
            assert m["capabilities"]["reasoning"] is True

    async def test_filter_function_calling(self, client: httpx.AsyncClient) -> None:
        """function_calling=true filter works."""
        resp = await client.get("/models?function_calling=true")
        data = resp.json()
        for m in data["models"]:
            assert m["capabilities"]["function_calling"] is True

    async def test_filter_streaming(self, client: httpx.AsyncClient) -> None:
        """streaming=true filter works."""
        resp = await client.get("/models?streaming=true")
        data = resp.json()
        for m in data["models"]:
            assert m["capabilities"]["streaming"] is True

    async def test_filter_structured_output(self, client: httpx.AsyncClient) -> None:
        """structured_output=true filter works."""
        resp = await client.get("/models?structured_output=true")
        data = resp.json()
        for m in data["models"]:
            assert m["capabilities"]["structured_output"] is True

    async def test_filter_json_mode(self, client: httpx.AsyncClient) -> None:
        """json_mode=true filter works."""
        resp = await client.get("/models?json_mode=true")
        data = resp.json()
        for m in data["models"]:
            assert m["capabilities"]["json_mode"] is True

    async def test_filter_family(self, client: httpx.AsyncClient) -> None:
        """family filter returns only matching models."""
        resp = await client.get("/models?family=gpt-5")
        data = resp.json()
        assert data["count"] > 0
        for m in data["models"]:
            assert m["family"] == "gpt-5"

    async def test_filter_family_gpt52(self, client: httpx.AsyncClient) -> None:
        """family=gpt-5.2 returns only GPT-5.2 models."""
        resp = await client.get("/models?family=gpt-5.2")
        data = resp.json()
        assert data["count"] == 1
        assert data["models"][0]["id"] == "gpt-5.2"

    async def test_filter_min_context(self, client: httpx.AsyncClient) -> None:
        """min_context filters correctly."""
        resp = await client.get("/models?min_context=500000")
        data = resp.json()
        for m in data["models"]:
            assert m["context_window"] >= 500000

    async def test_filter_max_input_price(self, client: httpx.AsyncClient) -> None:
        """max_input_price filters correctly."""
        resp = await client.get("/models?max_input_price=1.0")
        data = resp.json()
        for m in data["models"]:
            assert m["pricing"]["input_price_per_1m"] <= 1.0

    async def test_filter_max_output_price(self, client: httpx.AsyncClient) -> None:
        """max_output_price filters correctly."""
        resp = await client.get("/models?max_output_price=5.0")
        data = resp.json()
        for m in data["models"]:
            assert m["pricing"]["output_price_per_1m"] <= 5.0

    async def test_combined_filters(self, client: httpx.AsyncClient) -> None:
        """Multiple filters combine correctly."""
        resp = await client.get("/models?vision=true&reasoning=true")
        data = resp.json()
        for m in data["models"]:
            assert m["capabilities"]["vision"] is True
            assert m["capabilities"]["reasoning"] is True

    async def test_sort_by_name_asc(self, client: httpx.AsyncClient) -> None:
        """Default sort is by name ascending."""
        resp = await client.get("/models?sort=name&order=asc")
        data = resp.json()
        names = [m["name"].lower() for m in data["models"]]
        assert names == sorted(names)

    async def test_sort_by_name_desc(self, client: httpx.AsyncClient) -> None:
        """Sort by name descending."""
        resp = await client.get("/models?sort=name&order=desc")
        data = resp.json()
        names = [m["name"].lower() for m in data["models"]]
        assert names == sorted(names, reverse=True)

    async def test_sort_by_input_price_asc(self, client: httpx.AsyncClient) -> None:
        """Sort by input price ascending."""
        resp = await client.get("/models?sort=input_price&order=asc")
        data = resp.json()
        prices = [
            m["pricing"]["input_price_per_1m"]
            for m in data["models"]
            if m["pricing"]["input_price_per_1m"] is not None
        ]
        assert prices == sorted(prices)

    async def test_sort_by_output_price_desc(self, client: httpx.AsyncClient) -> None:
        """Sort by output price descending."""
        resp = await client.get("/models?sort=output_price&order=desc")
        data = resp.json()
        prices = [
            m["pricing"]["output_price_per_1m"]
            for m in data["models"]
            if m["pricing"]["output_price_per_1m"] is not None
        ]
        assert prices == sorted(prices, reverse=True)

    async def test_sort_by_context_window(self, client: httpx.AsyncClient) -> None:
        """Sort by context window."""
        resp = await client.get("/models?sort=context_window&order=desc")
        data = resp.json()
        windows = [
            m["context_window"]
            for m in data["models"]
            if m["context_window"] is not None
        ]
        assert windows == sorted(windows, reverse=True)

    async def test_sort_by_created(self, client: httpx.AsyncClient) -> None:
        """Sort by creation date."""
        resp = await client.get("/models?sort=created&order=desc")
        assert resp.status_code == 200

    async def test_search_q(self, client: httpx.AsyncClient) -> None:
        """Free-text search matches on id and name."""
        resp = await client.get("/models?q=gpt-5")
        data = resp.json()
        assert data["count"] > 0
        for m in data["models"]:
            assert "gpt-5" in m["id"].lower() or "gpt-5" in m["name"].lower()

    async def test_search_q_mini(self, client: httpx.AsyncClient) -> None:
        """Search for 'mini' matches mini models."""
        resp = await client.get("/models?q=mini")
        data = resp.json()
        assert data["count"] > 0
        for m in data["models"]:
            assert "mini" in m["id"].lower() or "mini" in m["name"].lower()

    async def test_search_case_insensitive(self, client: httpx.AsyncClient) -> None:
        """Search is case-insensitive."""
        resp1 = await client.get("/models?q=GPT")
        resp2 = await client.get("/models?q=gpt")
        assert resp1.json()["count"] == resp2.json()["count"]

    async def test_etag_returned(self, client: httpx.AsyncClient) -> None:
        """Response includes ETag header."""
        resp = await client.get("/models")
        assert "etag" in resp.headers

    async def test_etag_304(self, client: httpx.AsyncClient) -> None:
        """If-None-Match with matching ETag returns 304."""
        resp1 = await client.get("/models")
        etag = resp1.headers["etag"]

        resp2 = await client.get("/models", headers={"If-None-Match": etag})
        assert resp2.status_code == 304

    async def test_response_time_header(self, client: httpx.AsyncClient) -> None:
        """Response includes X-Response-Time header."""
        resp = await client.get("/models")
        assert "x-response-time" in resp.headers


class TestGetModel:
    """Tests for GET /models/{model_id}."""

    async def test_get_existing_model(self, client: httpx.AsyncClient) -> None:
        """Returns model data for valid ID."""
        resp = await client.get("/models/gpt-5.2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "gpt-5.2"
        assert data["name"] == "GPT-5.2"
        assert data["family"] == "gpt-5.2"
        assert data["context_window"] == 400000

    async def test_get_nonexistent_model(self, client: httpx.AsyncClient) -> None:
        """Returns 404 for unknown model ID."""
        resp = await client.get("/models/nonexistent")
        assert resp.status_code == 404
