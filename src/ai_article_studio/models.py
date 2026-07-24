"""SQLAlchemy persistence models for article tasks and agent execution logs."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class Article(Base):
    __tablename__ = "article"

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    topic: Mapped[str] = mapped_column(String(500))
    style: Mapped[str] = mapped_column(String(64), default="TECH")
    model_config: Mapped[str] = mapped_column(Text, default="{}")
    phase: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(64))
    title_options: Mapped[str | None] = mapped_column(Text, nullable=True)
    selected_title: Mapped[str | None] = mapped_column(Text, nullable=True)
    outline: Mapped[str | None] = mapped_column(Text, nullable=True)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    full_content: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_prompts: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_results: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    quality_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    review_report: Mapped[str | None] = mapped_column(Text, nullable=True)
    review_round: Mapped[int] = mapped_column(Integer, default=0)
    supervisor_action: Mapped[str | None] = mapped_column(String(64), nullable=True)
    supervisor_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    execution_mode: Mapped[str | None] = mapped_column(String(32), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_deleted: Mapped[int] = mapped_column(Integer, default=0)
    create_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    update_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class AgentLog(Base):
    __tablename__ = "agent_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[str] = mapped_column(String(64), index=True)
    agent_name: Mapped[str] = mapped_column(String(128))
    input_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32))
    cost_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    create_time: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
