"""Fast JSON serialization using orjson."""

from typing import Any

import orjson
from starlette.responses import JSONResponse


class ORJSONResponse(JSONResponse):
    """High-performance JSON response using orjson.

    3-10x faster than stdlib json serialization.
    """

    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        """Render content to JSON bytes using orjson."""
        return orjson.dumps(
            content,
            option=orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY,
        )
