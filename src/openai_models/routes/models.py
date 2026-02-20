"""Model listing and detail endpoints."""

import hashlib
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response

from openai_models.dependencies import StoreDep
from openai_models.models import (
    ModelFilterParams,
    ModelsListResponse,
    OpenAIModel,
    SortField,
    SortOrder,
)
from openai_models.serialization import ORJSONResponse

router = APIRouter()


def _get_filter_params(
    vision: bool | None = Query(default=None),
    reasoning: bool | None = Query(default=None),
    function_calling: bool | None = Query(default=None),
    structured_output: bool | None = Query(default=None),
    streaming: bool | None = Query(default=None),
    fine_tuning: bool | None = Query(default=None),
    logprobs: bool | None = Query(default=None),
    json_mode: bool | None = Query(default=None),
    distillation: bool | None = Query(default=None),
    predicted_outputs: bool | None = Query(default=None),
    family: str | None = Query(default=None),
    provider: str | None = Query(default=None),
    include_deprecated: bool = Query(default=False),
    min_context: int | None = Query(default=None),
    max_input_price: float | None = Query(default=None),
    max_output_price: float | None = Query(default=None),
    sort: SortField = Query(default=SortField.NAME),
    order: SortOrder = Query(default=SortOrder.ASC),
    q: str | None = Query(default=None),
) -> ModelFilterParams:
    """Parse and validate all query parameters."""
    return ModelFilterParams(
        vision=vision,
        reasoning=reasoning,
        function_calling=function_calling,
        structured_output=structured_output,
        streaming=streaming,
        fine_tuning=fine_tuning,
        logprobs=logprobs,
        json_mode=json_mode,
        distillation=distillation,
        predicted_outputs=predicted_outputs,
        family=family,
        provider=provider,
        include_deprecated=include_deprecated,
        min_context=min_context,
        max_input_price=max_input_price,
        max_output_price=max_output_price,
        sort=sort,
        order=order,
        q=q,
    )


FilterDep = Annotated[ModelFilterParams, Depends(_get_filter_params)]


@router.get("/models", response_model=ModelsListResponse)
async def list_models(
    store: StoreDep,
    params: FilterDep,
    response: Response,
    if_none_match: str | None = Header(default=None),
) -> ORJSONResponse | Response:
    """List models with filtering, sorting, and search."""
    snapshot = store.get_snapshot()

    # ETag based on last_refreshed
    etag = _compute_etag(snapshot.last_refreshed.isoformat() if snapshot.models else "")
    if if_none_match and if_none_match == etag:
        return Response(status_code=304)

    # Start from appropriate base list
    if params.include_deprecated:
        models = list(snapshot.models_list)
    else:
        models = list(snapshot.non_deprecated)

    # Apply capability filters
    capability_filters: dict[str, bool | None] = {
        "vision": params.vision,
        "reasoning": params.reasoning,
        "function_calling": params.function_calling,
        "structured_output": params.structured_output,
        "streaming": params.streaming,
        "fine_tuning": params.fine_tuning,
        "logprobs": params.logprobs,
        "json_mode": params.json_mode,
        "distillation": params.distillation,
        "predicted_outputs": params.predicted_outputs,
    }

    for attr, value in capability_filters.items():
        if value is not None:
            models = [m for m in models if getattr(m.capabilities, attr) == value]

    # Apply family filter
    if params.family:
        models = [m for m in models if m.family == params.family]

    # Apply provider filter
    if params.provider:
        models = [m for m in models if m.provider == params.provider]

    # Apply numeric filters
    if params.min_context is not None:
        models = [
            m
            for m in models
            if m.context_window is not None and m.context_window >= params.min_context
        ]

    if params.max_input_price is not None:
        models = [
            m
            for m in models
            if m.pricing.input_price_per_1m is not None
            and m.pricing.input_price_per_1m <= params.max_input_price
        ]

    if params.max_output_price is not None:
        models = [
            m
            for m in models
            if m.pricing.output_price_per_1m is not None
            and m.pricing.output_price_per_1m <= params.max_output_price
        ]

    # Apply free-text search
    if params.q:
        q_lower = params.q.lower()
        models = [
            m
            for m in models
            if q_lower in m.id.lower()
            or q_lower in m.name.lower()
            or q_lower in m.provider.lower()
        ]

    # Apply sorting
    models = _sort_models(models, params.sort, params.order)

    result = ModelsListResponse(
        count=len(models),
        last_refreshed=snapshot.last_refreshed if snapshot.models else None,
        models=models,
    )

    return ORJSONResponse(
        content=result.model_dump(mode="json"),
        headers={"ETag": etag},
    )


@router.get("/models/{model_id}", response_model=OpenAIModel)
async def get_model(store: StoreDep, model_id: str) -> OpenAIModel:
    """Get a single model by ID."""
    model = store.get_by_id(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return model


def _sort_models(
    models: list[OpenAIModel], sort: SortField, order: SortOrder
) -> list[OpenAIModel]:
    """Sort models by the given field and order."""
    reverse = order == SortOrder.DESC

    def _key(m: OpenAIModel) -> tuple[int, float | str]:
        """Return sort key with null handling (nulls last)."""
        match sort:
            case SortField.INPUT_PRICE:
                val = m.pricing.input_price_per_1m
                return (0 if val is not None else 1, val if val is not None else 0.0)
            case SortField.OUTPUT_PRICE:
                val = m.pricing.output_price_per_1m
                return (0 if val is not None else 1, val if val is not None else 0.0)
            case SortField.CONTEXT_WINDOW:
                val = m.context_window
                fval = float(val) if val is not None else 0.0
                return (0 if val is not None else 1, fval)
            case SortField.CREATED:
                dt = m.created_at
                ts = dt.timestamp() if dt is not None else 0.0
                return (0 if dt is not None else 1, ts)
            case SortField.NAME:
                return (0, m.name.lower() or m.id.lower())

    return sorted(models, key=_key, reverse=reverse)


def _compute_etag(content: str) -> str:
    """Compute an ETag from content string."""
    return f'"{hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()}"'
