# OpenAI Model Catalog API

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-e92063.svg?logo=pydantic&logoColor=white)](https://docs.pydantic.dev)
[![Ruff](https://img.shields.io/badge/linting-ruff-261230.svg?logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)
[![mypy](https://img.shields.io/badge/type--checked-mypy-blue.svg)](https://mypy-lang.org/)
[![pytest](https://img.shields.io/badge/tests-pytest-0A9EDC.svg?logo=pytest&logoColor=white)](https://docs.pytest.org)
[![Coverage](https://img.shields.io/badge/coverage-86%25-brightgreen.svg)](.)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg?logo=docker&logoColor=white)](Dockerfile)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet.svg)](https://docs.astral.sh/uv/)

Production-ready REST API that scrapes OpenAI's model information and serves it with rich filtering, sorting, and search.

## Features

- Scrapes model data from OpenAI API, docs pages, and pricing pages
- Copy-on-write store for lock-free reads (sub-10ms p99 latency)
- ETag-based conditional responses (304 Not Modified)
- Rich filtering: vision, reasoning, function_calling, family, price, context window
- Sorting by price, context window, name, or creation date
- Free-text search across model IDs and names
- Auto-refresh on configurable interval
- SQLite persistence for crash recovery
- orjson serialization (3-10x faster than stdlib json)

## Quick Start

### With uv

```bash
# Install dependencies
uv sync --dev

# Create .env from example
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run dev server
uv run uvicorn openai_models.app:create_app --factory --reload --port 8000
```

### With Docker

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
docker compose up --build
```

The API will be available at `http://localhost:8000`.

## API Endpoints

### `GET /models`

List models with filtering, sorting, and search.

```bash
# All non-deprecated models
curl http://localhost:8000/models

# Vision + reasoning models
curl "http://localhost:8000/models?vision=true&reasoning=true"

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
| `vision` | bool | Vision/image input support |
| `reasoning` | bool | Reasoning capability |
| `function_calling` | bool | Function calling support |
| `structured_output` | bool | Structured output support |
| `streaming` | bool | Streaming support |
| `fine_tuning` | bool | Fine-tuning available |
| `family` | string | Model family (e.g., `gpt-5.2`, `o3`) |
| `include_deprecated` | bool | Include deprecated models (default: false) |
| `min_context` | int | Minimum context window |
| `max_input_price` | float | Max input price per 1M tokens |
| `max_output_price` | float | Max output price per 1M tokens |
| `sort` | enum | `name`, `input_price`, `output_price`, `context_window`, `created` |
| `order` | enum | `asc` or `desc` |
| `q` | string | Free-text search |

### `GET /models/{model_id}`

```bash
curl http://localhost:8000/models/gpt-5.2
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
| `OPENAI_API_KEY` | *(required)* | API key for /v1/models |
| `APP_ENV` | `production` | `development` / `production` / `testing` |
| `APP_PORT` | `8000` | Bind port |
| `LOG_LEVEL` | `info` | Logging level |
| `REFRESH_INTERVAL_MINUTES` | `60` | Auto-refresh interval (0 = disabled) |
| `HTTP_TIMEOUT` | `30` | Outbound request timeout (seconds) |
| `DB_PATH` | `data/models.db` | SQLite path (empty = in-memory only) |

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

- **Copy-on-write store**: Reads never lock. Writers build a new immutable snapshot and atomically swap the reference pointer.
- **Scraping pipeline**: API endpoint -> docs pages -> pricing page -> hardcoded fallback. Each source enriches the previous.
- **orjson + ORJSONResponse**: All JSON serialization uses orjson for 3-10x speedup.
- **uvloop**: 2-4x faster event loop for async I/O.
- **ETag caching**: Repeat clients get 304 responses, skipping serialization entirely.
