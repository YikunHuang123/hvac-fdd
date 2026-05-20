from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


def make_engine(
    url: str,
    pool_size: int = 5,
    max_overflow: int = 10,
) -> Engine:
    """Create a SQLAlchemy engine.

    SQLite in-memory URLs receive two special-case overrides:
    - StaticPool pins all requests to a single connection so every caller
      sees the same in-memory database (each new connection is a blank DB).
    - check_same_thread=False lets the connection be used from FastAPI's
      threadpool workers when a session is injected via dependency_overrides.
    Regular file-path SQLite and non-SQLite URLs use the normal QueuePool.
    """
    kwargs: dict = {"pool_pre_ping": True}
    url_str = str(url)
    if url_str == "sqlite:///:memory:" or url_str.startswith("sqlite:///:memory:"):
        kwargs["connect_args"] = {"check_same_thread": False}
        kwargs["poolclass"] = StaticPool
    elif url_str.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_size"] = pool_size
        kwargs["max_overflow"] = max_overflow
    return create_engine(url, **kwargs)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a session factory bound to the given engine."""
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)
