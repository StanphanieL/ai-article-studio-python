"""Database setup and request-scoped SQLAlchemy sessions."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from .config import get_settings


class Base(DeclarativeBase):
    """Base for application persistence models."""


def _ensure_sqlite_directory(database_url: str) -> None:
    if database_url.startswith("sqlite:///") and not database_url.endswith(":memory:"):
        Path(database_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)


def create_app_engine(database_url: str | None = None):
    """Create the engine lazily so tests can supply isolated databases."""

    url = database_url or get_settings().database_url
    _ensure_sqlite_directory(url)
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args)


def create_session_factory(database_url: str) -> sessionmaker[Session]:
    """Create and initialize an isolated session factory for an application instance."""

    app_engine = create_app_engine(database_url)
    from . import models  # noqa: F401 - registers model metadata

    Base.metadata.create_all(bind=app_engine)
    _migrate_article_execution_mode(app_engine)
    return sessionmaker(
        bind=app_engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )


def _migrate_article_execution_mode(app_engine) -> None:
    """Add the one-click retry marker for databases created before this release.

    This project intentionally has a lightweight local schema setup instead of
    Alembic.  The statement is portable across the supported SQLite and
    PostgreSQL targets, and is guarded by inspection so fresh schemas are
    untouched.
    """

    inspector = inspect(app_engine)
    if "article" not in inspector.get_table_names():
        return
    existing = {column["name"] for column in inspector.get_columns("article")}
    if "execution_mode" in existing:
        return
    with app_engine.begin() as connection:
        connection.execute(text("ALTER TABLE article ADD COLUMN execution_mode VARCHAR(32)"))


engine = create_app_engine()
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)


def init_database() -> None:
    """Create the single-user local schema on application startup."""

    from . import models  # noqa: F401 - registers model metadata

    Base.metadata.create_all(bind=engine)
    _migrate_article_execution_mode(engine)


def get_session() -> Generator[Session]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
