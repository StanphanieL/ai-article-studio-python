"""Private persistence, sanitized logs, and task-owned asset cleanup.

Transaction boundaries and locking remain identical to the compatibility
facade; filesystem cleanup still runs only after the deletion commit.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .domain import WorkflowValidationError, redact_sensitive_text
from .models import AgentLog, Article

if TYPE_CHECKING:
    from .workflow import ArticleWorkflow


class _WorkflowPersistence:
    def __init__(self, workflow: ArticleWorkflow) -> None:
        self._workflow = workflow

    def permanently_delete(self, task_id: str, *, include_legacy_soft_deleted: bool = False) -> bool:
        """Physically remove one article, its logs, and its task-owned local assets.

        Callers must first stop or mark any active execution. The per-task lock
        prevents a worker from committing a result after the records are gone.
        """
        workflow = self._workflow

        with workflow.task_write_lock(task_id):
            with workflow.sessions() as session:
                article = (
                    session.scalar(select(Article).where(Article.task_id == task_id))
                    if include_legacy_soft_deleted
                    else workflow._get_article(session, task_id)
                )
                if not article:
                    raise WorkflowValidationError("文章任务不存在")
                local_paths = workflow._local_image_paths(article.image_results)
                referenced_paths = workflow._referenced_local_image_paths(session, excluding_task_id=task_id)
                session.execute(delete(AgentLog).where(AgentLog.task_id == task_id))
                session.delete(article)
                session.commit()
            workflow._delete_unreferenced_local_assets(task_id, local_paths, referenced_paths)
            return True

    def logs(self, task_id: str) -> list[AgentLog]:
        workflow = self._workflow
        with workflow.sessions() as session:
            logs = list(session.scalars(select(AgentLog).where(AgentLog.task_id == task_id).order_by(AgentLog.create_time.desc())))
            for log in logs:
                session.expunge(log)
            return logs

    def update(self, task_id: str, **values: object) -> Article:
        workflow = self._workflow
        with workflow.task_write_lock(task_id):
            with workflow.sessions() as session:
                article = workflow._get_article(session, task_id)
                workflow._ensure_article_writable(article)
                for field, value in values.items():
                    setattr(article, field, value)
                article.update_time = workflow._clock()
                session.commit()
                session.refresh(article)
                session.expunge(article)
                return article

    def update_downstream(self, task_id: str, **values: object) -> Article:
        workflow = self._workflow
        values.update({"image_prompts": None, "image_results": None, "final_markdown": None, "review_report": None, "quality_score": None})
        return workflow._update(task_id, **values)

    def record_decision(self, task_id: str, action: str, reason: str) -> Article:
        workflow = self._workflow
        return workflow._update(task_id, supervisor_action=action, supervisor_reason=reason)

    def mark_current_action(self, task_id: str, action: str) -> Article:
        """Persist the actual sub-step before invoking a provider.

        A one-click job is an orchestration mode, not a retry target.  Keeping
        the concrete operation here means an exception can be shown honestly
        and a normal step retry still has an exact provider-aware action.
        """
        workflow = self._workflow

        return workflow._update(
            task_id,
            supervisor_action=action,
            supervisor_reason=f"正在执行：{workflow.action_label(action)}",
        )

    def referenced_local_image_paths(self, session: Session, excluding_task_id: str) -> set[Path]:
        workflow = self._workflow
        referenced: set[Path] = set()
        rows = session.scalars(select(Article.image_results).where(Article.task_id != excluding_task_id))
        for raw_image_results in rows:
            referenced.update(workflow._local_image_paths(raw_image_results))
        return referenced

    def delete_unreferenced_local_assets(self, task_id: str, local_paths: set[Path], referenced_paths: set[Path]) -> None:
        """Clean this task's generated files without touching shared/manual assets."""
        workflow = self._workflow

        for path in local_paths - referenced_paths:
            path.unlink(missing_ok=True)

        root = workflow.settings.upload_dir.resolve()
        task_dir = (root / task_id).resolve()
        if not task_dir.is_relative_to(root) or not task_dir.is_dir():
            return
        for path in sorted(task_dir.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if path.is_file() and path.resolve() not in referenced_paths:
                path.unlink(missing_ok=True)
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
        try:
            task_dir.rmdir()
        except OSError:
            pass



def get_article(session: Session, task_id: str) -> Article:
    article = session.scalar(select(Article).where(Article.task_id == task_id, Article.is_deleted == 0))
    if not article:
        raise WorkflowValidationError("文章任务不存在")
    return article


def record_log(
    session: Session,
    task_id: str,
    agent_name: str,
    input_text: str,
    output_text: str | None,
    started: float,
    error: Exception | None = None,
) -> None:
    session.add(
        AgentLog(
            task_id=task_id,
            agent_name=agent_name,
            input_text=redact_sensitive_text(input_text)[:4000],
            output_text=redact_sensitive_text(output_text or "")[:8000] or None,
            status="FAILED" if error else "SUCCESS",
            cost_ms=int((time.perf_counter() - started) * 1000),
            error_message=safe_error_message(error)[:1000] if error else None,
        )
    )


def local_image_paths(raw_image_results: str | None, upload_dir: Path) -> set[Path]:
    """Resolve only app-hosted image URLs; never touch remote supplier URLs."""

    try:
        results = json.loads(raw_image_results or "[]")
    except json.JSONDecodeError:
        return set()
    if not isinstance(results, list):
        return set()

    root = upload_dir.resolve()
    paths: set[Path] = set()
    for result in results:
        if not isinstance(result, dict):
            continue
        image_url = result.get("imageUrl")
        if not isinstance(image_url, str) or not image_url.startswith("/uploads/images/"):
            continue
        relative = image_url.removeprefix("/uploads/images/")
        if not relative:
            continue
        candidate = (root / relative).resolve()
        if candidate.is_relative_to(root):
            paths.add(candidate)
    return paths


def safe_error_message(error: Exception) -> str:
    """Return diagnostics safe for persistence and page/API delivery."""
    return redact_sensitive_text(str(error))

