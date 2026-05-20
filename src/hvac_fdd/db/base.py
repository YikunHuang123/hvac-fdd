from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


def make_engine(
    url: str,
    pool_size: int = 5,
    max_overflow: int = 10,
) -> Engine:
    """Create a SQLAlchemy engine.

    Pool sizing parameters are omitted for SQLite, which uses a single-file
    connection model incompatible with connection pool sizing.
    """
    kwargs: dict = {"pool_pre_ping": True}
    if not str(url).startswith("sqlite"):
        kwargs["pool_size"] = pool_size
        kwargs["max_overflow"] = max_overflow
    return create_engine(url, **kwargs)


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a session factory bound to the given engine."""
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)
