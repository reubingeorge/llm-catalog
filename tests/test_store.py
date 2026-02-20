"""Tests for the copy-on-write ModelStore."""

import asyncio
import tempfile
from pathlib import Path

from openai_models.models import (
    OpenAIModel,
)
from openai_models.store import ModelSnapshot, ModelStore


class TestModelSnapshot:
    """Tests for the frozen ModelSnapshot dataclass."""

    def test_empty_snapshot(self) -> None:
        """Empty snapshot has no models."""
        snap = ModelSnapshot()
        assert len(snap.models) == 0
        assert len(snap.models_list) == 0
        assert len(snap.non_deprecated) == 0
        assert len(snap.by_family) == 0


class TestModelStore:
    """Tests for the ModelStore."""

    async def test_get_snapshot_returns_data(
        self, store: ModelStore, test_models: list[OpenAIModel]
    ) -> None:
        """get_snapshot returns the current snapshot with all models."""
        snapshot = store.get_snapshot()
        assert len(snapshot.models) == len(test_models)

    async def test_get_by_id_found(self, store: ModelStore) -> None:
        """get_by_id returns a model when it exists."""
        model = store.get_by_id("gpt-5.2")
        assert model is not None
        assert model.id == "gpt-5.2"
        assert model.name == "GPT-5.2"

    async def test_get_by_id_not_found(self, store: ModelStore) -> None:
        """get_by_id returns None for unknown model."""
        model = store.get_by_id("nonexistent-model")
        assert model is None

    async def test_replace_all_swaps_snapshot(self, store: ModelStore) -> None:
        """replace_all atomically replaces the snapshot."""
        old_snapshot = store.get_snapshot()
        new_models = [OpenAIModel(id="test-model", name="Test Model", family="test")]
        await store.replace_all(new_models)
        new_snapshot = store.get_snapshot()
        assert new_snapshot is not old_snapshot
        assert len(new_snapshot.models) == 1
        assert "test-model" in new_snapshot.models

    async def test_pre_computed_non_deprecated(self, store: ModelStore) -> None:
        """non_deprecated list excludes deprecated models."""
        snapshot = store.get_snapshot()
        deprecated_ids = {m.id for m in snapshot.models_list if m.deprecated}
        non_deprecated_ids = {m.id for m in snapshot.non_deprecated}
        assert not deprecated_ids & non_deprecated_ids
        assert "gpt-3.5-turbo" not in non_deprecated_ids

    async def test_pre_computed_by_family(self, store: ModelStore) -> None:
        """by_family groups models correctly."""
        snapshot = store.get_snapshot()
        assert "gpt-5" in snapshot.by_family
        gpt5_ids = {m.id for m in snapshot.by_family["gpt-5"]}
        assert "gpt-5" in gpt5_ids
        assert "gpt-5-mini" in gpt5_ids

    async def test_models_list_sorted_by_name(self, store: ModelStore) -> None:
        """models_list is pre-sorted by name."""
        snapshot = store.get_snapshot()
        names = [m.name.lower() or m.id.lower() for m in snapshot.models_list]
        assert names == sorted(names)

    async def test_concurrent_reads_during_write(
        self, test_models: list[OpenAIModel]
    ) -> None:
        """Concurrent reads don't block during a write."""
        store = ModelStore(db_path=None)
        await store.replace_all(test_models)

        read_results: list[int] = []
        write_done = asyncio.Event()

        async def reader() -> None:
            for _ in range(10):
                snap = store.get_snapshot()
                read_results.append(len(snap.models))
                await asyncio.sleep(0.001)

        async def writer() -> None:
            new_models = [OpenAIModel(id="new", name="New", family="new")]
            await store.replace_all(new_models)
            write_done.set()

        await asyncio.gather(reader(), reader(), reader(), writer())
        # All reads should have succeeded without error
        assert len(read_results) == 30
        # Each read got a consistent snapshot (either old or new count)
        for count in read_results:
            assert count in (len(test_models), 1)

    async def test_snapshot_immutability(self, store: ModelStore) -> None:
        """Old snapshot references remain valid after a new write."""
        old_snap = store.get_snapshot()
        old_count = len(old_snap.models)

        await store.replace_all([OpenAIModel(id="x", name="X", family="x")])

        # Old snapshot is unmodified
        assert len(old_snap.models) == old_count
        # New snapshot is different
        assert len(store.get_snapshot().models) == 1

    async def test_sqlite_round_trip(self, test_models: list[OpenAIModel]) -> None:
        """Models survive a save → clear → load cycle via SQLite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store1 = ModelStore(db_path=db_path)
            await store1.init_db()
            await store1.replace_all(test_models)

            # New store loading from same DB
            store2 = ModelStore(db_path=db_path)
            loaded = await store2.load_from_db()
            assert loaded is True

            snap = store2.get_snapshot()
            assert len(snap.models) == len(test_models)

            # Verify fields survived
            gpt52 = snap.models.get("gpt-5.2")
            assert gpt52 is not None
            assert gpt52.name == "GPT-5.2"
            assert gpt52.context_window == 400_000
            assert gpt52.pricing.input_price_per_1m == 1.75
            assert gpt52.capabilities.vision is True

    async def test_load_from_db_no_file(self, tmp_path: Path) -> None:
        """load_from_db returns False when DB file doesn't exist."""
        store = ModelStore(db_path=tmp_path / "nonexistent.db")
        result = await store.load_from_db()
        assert result is False

    async def test_load_from_db_disabled(self) -> None:
        """load_from_db returns False when db_path is None."""
        store = ModelStore(db_path=None)
        result = await store.load_from_db()
        assert result is False

    async def test_get_last_refreshed_empty(self) -> None:
        """get_last_refreshed returns None for empty store."""
        store = ModelStore(db_path=None)
        assert store.get_last_refreshed() is None

    async def test_get_last_refreshed_populated(self, store: ModelStore) -> None:
        """get_last_refreshed returns a datetime for populated store."""
        assert store.get_last_refreshed() is not None
