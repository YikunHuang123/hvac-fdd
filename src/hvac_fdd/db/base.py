from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

class Base(DeclarativeBase):
    pass

def make_engine(url: str):
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)

def make_session_factory(engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)
