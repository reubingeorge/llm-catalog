"""Pydantic schemas for OpenAI model data and API responses."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class SortField(StrEnum):
    """Fields available for sorting model results."""

    INPUT_PRICE = "input_price"
    OUTPUT_PRICE = "output_price"
    CONTEXT_WINDOW = "context_window"
    NAME = "name"
    CREATED = "created"


class SortOrder(StrEnum):
    """Sort direction."""

    ASC = "asc"
    DESC = "desc"


class ModelCapabilities(BaseModel):
    """Feature flags for a model."""

    model_config = ConfigDict(frozen=True)

    vision: bool = False
    reasoning: bool = False
    function_calling: bool = False
    structured_output: bool = False
    streaming: bool = False
    fine_tuning: bool = False
    logprobs: bool = False
    json_mode: bool = False
    distillation: bool = False
    predicted_outputs: bool = False


class ModelPricing(BaseModel):
    """Pricing information per 1M tokens (USD)."""

    model_config = ConfigDict(frozen=True)

    input_price_per_1m: float | None = None
    output_price_per_1m: float | None = None
    cached_input_price_per_1m: float | None = None


class OpenAIModel(BaseModel):
    """Full representation of an OpenAI model."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str = ""
    family: str = ""
    provider: str = "openai"
    description: str = ""
    context_window: int | None = None
    max_output_tokens: int | None = None
    knowledge_cutoff: str | None = None
    deprecated: bool = False
    deprecation_date: str | None = None
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    pricing: ModelPricing = Field(default_factory=ModelPricing)
    endpoints: list[str] = Field(default_factory=list)
    rate_limits: dict[str, object] = Field(default_factory=dict)
    created_at: datetime | None = None
    scraped_at: datetime | None = None


class ModelFilterParams(BaseModel):
    """Query parameters for filtering models."""

    vision: bool | None = None
    reasoning: bool | None = None
    function_calling: bool | None = None
    structured_output: bool | None = None
    streaming: bool | None = None
    fine_tuning: bool | None = None
    logprobs: bool | None = None
    json_mode: bool | None = None
    distillation: bool | None = None
    predicted_outputs: bool | None = None
    family: str | None = None
    provider: str | None = None
    include_deprecated: bool = False
    min_context: int | None = None
    max_input_price: float | None = None
    max_output_price: float | None = None
    sort: SortField = SortField.NAME
    order: SortOrder = SortOrder.ASC
    q: str | None = None


class ModelsListResponse(BaseModel):
    """Response for GET /models."""

    count: int
    last_refreshed: datetime | None
    models: list[OpenAIModel]


class RefreshRequest(BaseModel):
    """Request body for POST /refresh."""

    probe_capabilities: bool = False


class RefreshResponse(BaseModel):
    """Response for POST /refresh."""

    status: str
    models_found: int
    duration_seconds: float
    refreshed_at: datetime


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str
    models_loaded: int
    last_refreshed: datetime | None
    uptime_seconds: float
    version: str
