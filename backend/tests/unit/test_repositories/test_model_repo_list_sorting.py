from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import select

from app.db.models import ModelMapping as ModelMappingORM
from app.domain.model import ModelMappingCreate
from app.repositories.sqlalchemy.model_repo import SQLAlchemyModelRepository


@pytest_asyncio.fixture
async def model_repo(db_session):
    return SQLAlchemyModelRepository(db_session)


async def _set_created_at(db_session, requested_model: str, created_at: datetime) -> None:
    result = await db_session.execute(
        select(ModelMappingORM).where(ModelMappingORM.requested_model == requested_model)
    )
    entity = result.scalar_one()
    entity.created_at = created_at.replace(tzinfo=None)
    await db_session.commit()


@pytest.mark.asyncio
class TestModelRepoListSorting:
    async def test_get_all_mappings_defaults_to_latest_created_first(self, model_repo, db_session):
        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for offset, requested_model in enumerate(["alpha", "charlie", "bravo"]):
            await model_repo.create_mapping(ModelMappingCreate(requested_model=requested_model))
            await _set_created_at(db_session, requested_model, base_time + timedelta(minutes=offset))

        items, total = await model_repo.get_all_mappings(page=1, page_size=20)

        assert total == 3
        assert [item.requested_model for item in items] == ["bravo", "charlie", "alpha"]

    async def test_get_all_mappings_supports_requested_model_ascending_sort(
        self, model_repo, db_session
    ):
        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for offset, requested_model in enumerate(["charlie", "bravo", "alpha"]):
            await model_repo.create_mapping(ModelMappingCreate(requested_model=requested_model))
            await _set_created_at(db_session, requested_model, base_time + timedelta(minutes=offset))

        items, total = await model_repo.get_all_mappings(
            page=1,
            page_size=20,
            sort_by="requested_model_asc",
        )

        assert total == 3
        assert [item.requested_model for item in items] == ["alpha", "bravo", "charlie"]

    async def test_get_all_mappings_supports_requested_model_descending_sort(
        self, model_repo, db_session
    ):
        base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for offset, requested_model in enumerate(["alpha", "bravo", "charlie"]):
            await model_repo.create_mapping(ModelMappingCreate(requested_model=requested_model))
            await _set_created_at(db_session, requested_model, base_time + timedelta(minutes=offset))

        items, total = await model_repo.get_all_mappings(
            page=1,
            page_size=20,
            sort_by="requested_model_desc",
        )

        assert total == 3
        assert [item.requested_model for item in items] == ["charlie", "bravo", "alpha"]
