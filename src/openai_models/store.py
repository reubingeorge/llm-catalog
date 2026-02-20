"""Copy-on-write model store with immutable snapshots.

Reads never acquire a lock. Writers build a new snapshot and atomically
swap the reference. Python's GIL guarantees reference assignment is atomic.
"""

import asyncio
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import structlog

from openai_models.models import (
    OpenAIModel,
)

logger = structlog.stdlib.get_logger()


@dataclass(frozen=True)
class ModelSnapshot:
    """Immutable snapshot of all model data.

    Pre-computed indexes enable O(1) lookups without any locking.
    """

    models: dict[str, OpenAIModel] = field(default_factory=dict)
    models_list: list[OpenAIModel] = field(default_factory=list)
    non_deprecated: list[OpenAIModel] = field(default_factory=list)
    by_family: dict[str, list[OpenAIModel]] = field(default_factory=dict)
    last_refreshed: datetime = field(default_factory=lambda: datetime.now(tz=UTC))


_EMPTY_SNAPSHOT = ModelSnapshot()


class ModelStore:
    """Thread-safe, lock-free-read model store.

    Uses copy-on-write pattern: reads return an atomic snapshot reference,
    writes build a new snapshot and swap the pointer.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Initialize the store with optional SQLite persistence."""
        self._snapshot: ModelSnapshot = _EMPTY_SNAPSHOT
        self._refresh_lock = asyncio.Lock()
        self._db_path = db_path

    def get_snapshot(self) -> ModelSnapshot:
        """Return the current snapshot. No lock needed."""
        return self._snapshot

    def get_by_id(self, model_id: str) -> OpenAIModel | None:
        """Look up a model by ID. O(1) dict lookup, no lock."""
        return self._snapshot.models.get(model_id)

    @property
    def refresh_lock(self) -> asyncio.Lock:
        """Expose the refresh lock for the refresh route."""
        return self._refresh_lock

    async def replace_all(self, models: list[OpenAIModel]) -> None:
        """Replace all models atomically under the write lock."""
        async with self._refresh_lock:
            new_snapshot = self._build_snapshot(models)
            self._snapshot = new_snapshot  # Atomic pointer swap
            await self._persist(new_snapshot)

    async def replace_all_unlocked(self, models: list[OpenAIModel]) -> None:
        """Replace all models when the caller already holds the lock."""
        new_snapshot = self._build_snapshot(models)
        self._snapshot = new_snapshot
        await self._persist(new_snapshot)

    @staticmethod
    def _build_snapshot(models: list[OpenAIModel]) -> ModelSnapshot:
        """Build a new immutable snapshot with pre-computed indexes."""
        models_dict: dict[str, OpenAIModel] = {m.id: m for m in models}
        models_list = sorted(models, key=lambda m: m.name or m.id)
        non_deprecated = [m for m in models_list if not m.deprecated]

        by_family: dict[str, list[OpenAIModel]] = defaultdict(list)
        for m in models_list:
            if m.family:
                by_family[m.family].append(m)

        return ModelSnapshot(
            models=models_dict,
            models_list=models_list,
            non_deprecated=non_deprecated,
            by_family=dict(by_family),
            last_refreshed=datetime.now(tz=UTC),
        )

    async def init_db(self) -> None:
        """Create the SQLite table if persistence is enabled."""
        if self._db_path is None:
            return

        import aiosqlite

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            await db.commit()

    async def load_from_db(self) -> bool:
        """Load cached model data from SQLite. Returns True if data was loaded."""
        if self._db_path is None or not self._db_path.exists():
            return False

        import aiosqlite

        try:
            async with aiosqlite.connect(self._db_path) as db:
                cursor = await db.execute("SELECT id, data FROM models")
                rows = await cursor.fetchall()

            if not rows:
                return False

            models: list[OpenAIModel] = []
            for _id, data_json in rows:
                data = json.loads(data_json)
                models.append(OpenAIModel.model_validate(data))

            self._snapshot = self._build_snapshot(models)
            await logger.ainfo(
                "loaded_models_from_db",
                count=len(models),
                db_path=str(self._db_path),
            )
            return True
        except Exception:
            await logger.awarning(
                "failed_to_load_from_db",
                db_path=str(self._db_path),
                exc_info=True,
            )
            return False

    async def _persist(self, snapshot: ModelSnapshot) -> None:
        """Persist the snapshot to SQLite."""
        if self._db_path is None:
            return

        import aiosqlite

        try:
            async with aiosqlite.connect(self._db_path) as db:
                await db.execute("DELETE FROM models")
                now = datetime.now(tz=UTC).isoformat()
                for model in snapshot.models_list:
                    await db.execute(
                        "INSERT INTO models (id, data, updated_at) VALUES (?, ?, ?)",
                        (model.id, model.model_dump_json(), now),
                    )
                await db.commit()
        except Exception:
            await logger.awarning(
                "failed_to_persist",
                db_path=str(self._db_path),
                exc_info=True,
            )

    def get_last_refreshed(self) -> datetime | None:
        """Return the last refresh time, or None if never refreshed."""
        snap = self._snapshot
        if not snap.models:
            return None
        return snap.last_refreshed
