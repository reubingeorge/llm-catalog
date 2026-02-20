# Stage 1: Builder
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy all project files needed for install
COPY pyproject.toml uv.lock ./
COPY src/ src/

# Sync production dependencies only
RUN uv sync --frozen --no-dev --no-editable

# Stage 2: Runtime
FROM python:3.12-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Copy venv and source from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

# Create non-root user and data directory
RUN useradd --system --no-create-home appuser && \
    mkdir -p /app/data && \
    chown appuser /app/data

USER appuser

EXPOSE 8000

ENV PATH="/app/.venv/bin:$PATH"

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["gunicorn", "openai_models.app:create_app", \
     "--factory", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", \
     "--bind", "0.0.0.0:8000", \
     "--keep-alive", "120", \
     "--timeout", "60", \
     "--graceful-timeout", "10", \
     "--access-logfile", "-"]
