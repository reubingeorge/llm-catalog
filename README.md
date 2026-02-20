# LLM Catalog API

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-e92063.svg?logo=pydantic&logoColor=white)](https://docs.pydantic.dev)
[![Ruff](https://img.shields.io/badge/linting-ruff-261230.svg?logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)
[![mypy](https://img.shields.io/badge/type--checked-mypy-blue.svg)](https://mypy-lang.org/)
[![pytest](https://img.shields.io/badge/tests-pytest-0A9EDC.svg?logo=pytest&logoColor=white)](https://docs.pytest.org)
[![Coverage](https://img.shields.io/badge/coverage-81%25-brightgreen.svg)](.)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg?logo=docker&logoColor=white)](Dockerfile)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet.svg)](https://docs.astral.sh/uv/)

Multi-provider REST API that catalogs LLM models from **OpenAI**, **Anthropic (Claude)**, and **Google (Gemini)** with rich filtering, sorting, and search.

## Features

- **Multi-provider**: Scrapes and serves models from OpenAI, Anthropic, and Google
- Copy-on-write store for lock-free reads (sub-10ms p99 latency)
- ETag-based conditional responses (304 Not Modified)
- Rich filtering: provider, vision, reasoning, function_calling, family, price, context window
- Sorting by price, context window, name, or creation date
- Free-text search across model IDs, names, and providers
- Auto-refresh on configurable interval
- SQLite persistence for crash recovery
- orjson serialization (3-10x faster than stdlib json)

## Quick Start

### With uv

```bash
# Install dependencies
uv sync --dev

# Create .env from example (optional for local dev)
cp .env.example .env
# Edit .env and add your API keys

# Run dev server
uv run uvicorn openai_models.app:create_app --factory --reload --port 8000
```

### With Docker

```bash
cp .env.example .env
# Edit .env and add your API keys
docker compose up --build
```

The API will be available at `http://localhost:8000`.

## API Endpoints

### `GET /models`

List models with filtering, sorting, and search.

```bash
# All non-deprecated models (all providers)
curl http://localhost:8000/models

# Only OpenAI models
curl "http://localhost:8000/models?provider=openai"

# Only Anthropic (Claude) models
curl "http://localhost:8000/models?provider=anthropic"

# Only Google (Gemini) models
curl "http://localhost:8000/models?provider=google"

# Vision + reasoning models across all providers
curl "http://localhost:8000/models?vision=true&reasoning=true"

# Claude models with vision support
curl "http://localhost:8000/models?provider=anthropic&vision=true"

# GPT-5 family, sorted by input price
curl "http://localhost:8000/models?family=gpt-5&sort=input_price&order=asc"

# Budget models under $1/1M input tokens
curl "http://localhost:8000/models?max_input_price=1.0"

# High-context models (500K+ tokens)
curl "http://localhost:8000/models?min_context=500000"

# Search for "mini" models
curl "http://localhost:8000/models?q=mini"

# Include deprecated models
curl "http://localhost:8000/models?include_deprecated=true"
```

**Query Parameters:**

| Param | Type | Description |
|---|---|---|
| `provider` | string | Filter by provider: `openai`, `anthropic`, `google` |
| `vision` | bool | Vision/image input support |
| `reasoning` | bool | Reasoning capability |
| `function_calling` | bool | Function calling support |
| `structured_output` | bool | Structured output support |
| `streaming` | bool | Streaming support |
| `fine_tuning` | bool | Fine-tuning available |
| `family` | string | Model family (e.g., `gpt-5.2`, `claude-sonnet`, `gemini-2.5`) |
| `include_deprecated` | bool | Include deprecated models (default: false) |
| `min_context` | int | Minimum context window |
| `max_input_price` | float | Max input price per 1M tokens |
| `max_output_price` | float | Max output price per 1M tokens |
| `sort` | enum | `name`, `input_price`, `output_price`, `context_window`, `created` |
| `order` | enum | `asc` or `desc` |
| `q` | string | Free-text search (matches id, name, and provider) |

### `GET /models/{model_id}`

```bash
curl http://localhost:8000/models/gpt-5.2
curl http://localhost:8000/models/claude-sonnet-4-5-20250929
curl http://localhost:8000/models/gemini-2.5-pro
```

### `POST /refresh`

```bash
curl -X POST http://localhost:8000/refresh
```

### `GET /health`

```bash
curl http://localhost:8000/health
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(optional)* | OpenAI API key |
| `ANTHROPIC_API_KEY` | *(optional)* | Anthropic API key |
| `GEMINI_API_KEY` | *(optional)* | Google Gemini API key |
| `APP_ENV` | `production` | `development` / `production` / `testing` |
| `APP_PORT` | `8000` | Bind port |
| `LOG_LEVEL` | `info` | Logging level |
| `REFRESH_INTERVAL_MINUTES` | `60` | Auto-refresh interval (0 = disabled) |
| `HTTP_TIMEOUT` | `30` | Outbound request timeout (seconds) |
| `DB_PATH` | `data/models.db` | SQLite path (empty = in-memory only) |

All API keys are optional. If a key is missing, that provider's live API is skipped and models are served from hardcoded fallback data.

## Development

```bash
# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/

# Type check
uv run mypy src/

# Test (requires 80% coverage)
uv run pytest
```

## Architecture

- **Multi-provider scraping**: OpenAI, Anthropic, and Gemini scrapers run concurrently. Each provider follows the same merge priority: API response > scraped pricing/docs > hardcoded fallback.
- **Copy-on-write store**: Reads never lock. Writers build a new immutable snapshot and atomically swap the reference pointer. Pre-computed indexes by family and provider.
- **orjson + ORJSONResponse**: All JSON serialization uses orjson for 3-10x speedup.
- **uvloop**: 2-4x faster event loop for async I/O.
- **ETag caching**: Repeat clients get 304 responses, skipping serialization entirely.
