"""Entrypoint for running with `python -m openai_models`."""

import uvicorn

from openai_models.config import get_settings


def main() -> None:
    """Run the application with uvicorn."""
    settings = get_settings()
    uvicorn.run(
        "openai_models.app:create_app",
        factory=True,
        host=settings.app_host,
        port=settings.app_port,
        loop="uvloop",
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
