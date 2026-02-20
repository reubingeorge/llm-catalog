"""Application configuration loaded from environment variables."""

from enum import StrEnum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    """Application environment."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str = Field(default="", description="OpenAI API key")
    app_env: Environment = Field(default=Environment.PRODUCTION)
    app_host: str = Field(default="0.0.0.0")  # noqa: S104
    app_port: int = Field(default=8000)
    log_level: str = Field(default="info")
    refresh_interval_minutes: int = Field(default=60)
    scrape_concurrency: int = Field(default=5)
    http_timeout: int = Field(default=30)
    db_path: str = Field(default="data/models.db")

    @property
    def db_file(self) -> Path | None:
        """Return the database file path, creating parent dirs if needed."""
        if not self.db_path:
            return None
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def is_testing(self) -> bool:
        """Check if running in test environment."""
        return self.app_env == Environment.TESTING


def get_settings() -> Settings:
    """Create a Settings instance."""
    return Settings()
