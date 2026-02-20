"""Shared test fixtures."""

import json
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport

from openai_models.config import Environment, Settings
from openai_models.logging import setup_logging
from openai_models.models import (
    ModelCapabilities,
    ModelPricing,
    OpenAIModel,
)
from openai_models.store import ModelStore

# Initialize logging for tests
setup_logging(Environment.TESTING, "debug")

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def test_settings() -> Settings:
    """Settings configured for testing."""
    return Settings(
        openai_api_key="sk-test-key-fake",
        app_env=Environment.TESTING,
        app_host="127.0.0.1",
        app_port=8000,
        log_level="debug",
        refresh_interval_minutes=0,
        scrape_concurrency=2,
        http_timeout=5,
        db_path="",
    )


def _make_test_models() -> list[OpenAIModel]:
    """Create a realistic set of test models."""
    now = datetime.now(tz=UTC)
    return [
        OpenAIModel(
            id="gpt-5.2",
            name="GPT-5.2",
            family="gpt-5.2",
            description="Most capable model",
            context_window=400_000,
            max_output_tokens=128_000,
            capabilities=ModelCapabilities(
                reasoning=True,
                vision=True,
                function_calling=True,
                structured_output=True,
                streaming=True,
                json_mode=True,
            ),
            pricing=ModelPricing(
                input_price_per_1m=1.75,
                output_price_per_1m=14.00,
                cached_input_price_per_1m=0.175,
            ),
            created_at=datetime(2026, 1, 15, tzinfo=UTC),
            scraped_at=now,
        ),
        OpenAIModel(
            id="gpt-5",
            name="GPT-5",
            family="gpt-5",
            description="Powerful model",
            context_window=400_000,
            max_output_tokens=128_000,
            capabilities=ModelCapabilities(
                reasoning=True,
                function_calling=True,
                structured_output=True,
                streaming=True,
                json_mode=True,
            ),
            pricing=ModelPricing(
                input_price_per_1m=1.25,
                output_price_per_1m=10.00,
                cached_input_price_per_1m=0.125,
            ),
            created_at=datetime(2025, 12, 1, tzinfo=UTC),
            scraped_at=now,
        ),
        OpenAIModel(
            id="gpt-5-mini",
            name="GPT-5 Mini",
            family="gpt-5",
            description="Smaller GPT-5",
            context_window=400_000,
            max_output_tokens=128_000,
            capabilities=ModelCapabilities(
                function_calling=True,
                structured_output=True,
                streaming=True,
                json_mode=True,
            ),
            pricing=ModelPricing(
                input_price_per_1m=0.25,
                output_price_per_1m=2.00,
                cached_input_price_per_1m=0.025,
            ),
            created_at=datetime(2025, 12, 1, tzinfo=UTC),
            scraped_at=now,
        ),
        OpenAIModel(
            id="gpt-4.1",
            name="GPT-4.1",
            family="gpt-4.1",
            description="High context model",
            context_window=1_048_000,
            max_output_tokens=32_000,
            capabilities=ModelCapabilities(
                vision=True,
                function_calling=True,
                structured_output=True,
                streaming=True,
                json_mode=True,
            ),
            pricing=ModelPricing(
                input_price_per_1m=2.00,
                output_price_per_1m=8.00,
                cached_input_price_per_1m=0.50,
            ),
            created_at=datetime(2025, 10, 1, tzinfo=UTC),
            scraped_at=now,
        ),
        OpenAIModel(
            id="o4-mini",
            name="o4-mini",
            family="o4",
            description="Fast reasoning",
            context_window=200_000,
            max_output_tokens=100_000,
            capabilities=ModelCapabilities(
                reasoning=True,
                vision=True,
                function_calling=True,
                structured_output=True,
                streaming=True,
                json_mode=True,
            ),
            pricing=ModelPricing(
                input_price_per_1m=1.10,
                output_price_per_1m=4.40,
                cached_input_price_per_1m=0.275,
            ),
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            scraped_at=now,
        ),
        OpenAIModel(
            id="o3",
            name="o3",
            family="o3",
            description="Advanced reasoning",
            context_window=200_000,
            max_output_tokens=100_000,
            capabilities=ModelCapabilities(
                reasoning=True,
                function_calling=True,
                structured_output=True,
                streaming=True,
                json_mode=True,
            ),
            pricing=ModelPricing(
                input_price_per_1m=2.00,
                output_price_per_1m=8.00,
                cached_input_price_per_1m=0.50,
            ),
            created_at=datetime(2025, 8, 1, tzinfo=UTC),
            scraped_at=now,
        ),
        OpenAIModel(
            id="gpt-4o",
            name="GPT-4o",
            family="gpt-4o",
            description="Multimodal model",
            context_window=128_000,
            max_output_tokens=16_000,
            capabilities=ModelCapabilities(
                vision=True,
                function_calling=True,
                structured_output=True,
                streaming=True,
                json_mode=True,
            ),
            pricing=ModelPricing(
                input_price_per_1m=2.50,
                output_price_per_1m=10.00,
                cached_input_price_per_1m=1.25,
            ),
            created_at=datetime(2025, 5, 1, tzinfo=UTC),
            scraped_at=now,
        ),
        OpenAIModel(
            id="gpt-4o-mini",
            name="GPT-4o Mini",
            family="gpt-4o",
            description="Small multimodal",
            context_window=128_000,
            max_output_tokens=16_000,
            capabilities=ModelCapabilities(
                vision=True,
                function_calling=True,
                structured_output=True,
                streaming=True,
                json_mode=True,
            ),
            pricing=ModelPricing(
                input_price_per_1m=0.15,
                output_price_per_1m=0.60,
                cached_input_price_per_1m=0.075,
            ),
            created_at=datetime(2025, 5, 1, tzinfo=UTC),
            scraped_at=now,
        ),
        OpenAIModel(
            id="gpt-oss-20b",
            name="GPT-OSS 20B",
            family="gpt-oss",
            description="Open-weight model",
            context_window=131_000,
            capabilities=ModelCapabilities(streaming=True),
            pricing=ModelPricing(
                input_price_per_1m=0.03,
                output_price_per_1m=0.14,
            ),
            created_at=datetime(2026, 1, 10, tzinfo=UTC),
            scraped_at=now,
        ),
        OpenAIModel(
            id="gpt-3.5-turbo",
            name="GPT-3.5 Turbo",
            family="gpt-3.5",
            description="Legacy model",
            context_window=16_000,
            max_output_tokens=4_000,
            deprecated=True,
            capabilities=ModelCapabilities(
                function_calling=True,
                streaming=True,
                json_mode=True,
            ),
            pricing=ModelPricing(
                input_price_per_1m=0.50,
                output_price_per_1m=1.50,
            ),
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
            scraped_at=now,
        ),
    ]


@pytest.fixture
def test_models() -> list[OpenAIModel]:
    """Return a set of test models."""
    return _make_test_models()


@pytest.fixture
async def store(test_models: list[OpenAIModel]) -> ModelStore:
    """A pre-populated model store."""
    s = ModelStore(db_path=None)
    await s.replace_all(test_models)
    return s


@pytest.fixture
async def app(test_settings: Settings, store: ModelStore) -> AsyncIterator[object]:
    """Create a test FastAPI app with pre-populated store."""
    from openai_models.app import create_app

    application = create_app(settings=test_settings)
    # Override lifespan by setting state directly
    application.state.store = store
    application.state.settings = test_settings
    application.state.start_time = time.monotonic()
    application.state.http_client = httpx.AsyncClient()

    yield application

    await application.state.http_client.aclose()


@pytest.fixture
async def client(app: object) -> AsyncIterator[httpx.AsyncClient]:
    """Async HTTP test client."""
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def api_response_json() -> dict[str, object]:
    """Load the mock OpenAI API response fixture."""
    path = FIXTURES_DIR / "openai_api_response.json"
    with open(path) as f:
        return json.load(f)  # type: ignore[no-any-return]


@pytest.fixture
def models_page_html() -> str:
    """Load the mock models page HTML fixture."""
    path = FIXTURES_DIR / "models_page.html"
    return path.read_text()


@pytest.fixture
def pricing_page_html() -> str:
    """Load the mock pricing page HTML fixture."""
    path = FIXTURES_DIR / "pricing_page.html"
    return path.read_text()
