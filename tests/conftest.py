import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from hvac_fdd.config import Settings, get_settings
from hvac_fdd.db.base import Base, make_engine, make_session_factory


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Override settings for the test session: in-memory SQLite + fixture data dirs."""
    return Settings(
        database_url="sqlite:///:memory:",
        lbnl_data_dir="tests/fixtures/data",
        processed_data_dir="tests/fixtures/processed",
        models_dir="tests/fixtures/models",
    )


@pytest.fixture(scope="session")
def engine(test_settings: Settings) -> Engine:
    eng = make_engine(test_settings.database_url)
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)


@pytest.fixture(scope="session")
def session_factory(engine: Engine) -> sessionmaker[Session]:
    return make_session_factory(engine)


@pytest.fixture
def db_session(session_factory: sessionmaker[Session]) -> Session:
    """Transactional session — rolls back after each test to keep state isolated."""
    session = session_factory()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def override_settings(test_settings: Settings, monkeypatch: pytest.MonkeyPatch):
    """Patch get_settings() so any code calling it receives test_settings."""
    monkeypatch.setattr("hvac_fdd.config.get_settings", lambda: test_settings)
    get_settings.cache_clear()
    yield test_settings
    get_settings.cache_clear()
