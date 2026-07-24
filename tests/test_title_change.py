from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from ai_article_studio.config import RuntimeMode, Settings
from ai_article_studio.db import create_session_factory
from ai_article_studio.llm import create_article_agent
from ai_article_studio.models import Article
from ai_article_studio.providers import create_image_service
from ai_article_studio.supervisor import ArticlePhase, ArticleStatus, SupervisorAction
from ai_article_studio.workflow import ArticleWorkflow


def _workflow(tmp_path: Path) -> ArticleWorkflow:
    settings = Settings(
        app_mode=RuntimeMode.MOCK,
        database_url=f"sqlite:///{tmp_path / 'articles.db'}",
        upload_dir=tmp_path / "images",
    )
    return ArticleWorkflow(
        create_session_factory(settings.database_url),
        settings,
        create_article_agent(settings),
        create_image_service(settings),
    )


def _seed_completed_article(workflow: ArticleWorkflow, *, selected_title: str | None = "原标题") -> tuple[str, str]:
    task_id = workflow.create("标题变更测试")
    with workflow.sessions() as session:
        article = session.scalar(select(Article).where(Article.task_id == task_id))
        assert article is not None
        model_config = article.model_config
        article.title_options = '[{"title": "原标题"}, {"title": "新标题"}]'
        article.selected_title = selected_title
        article.outline = '[{"heading": "旧大纲"}]'
        article.content = "# 原标题\n\n## 旧正文"
        article.full_content = "旧全文"
        article.image_prompts = '[{"imageTitle": "旧提示词"}]'
        article.image_results = '[{"imageUrl": "/uploads/images/old.png"}]'
        article.final_markdown = "# 原标题\n\n旧图文稿"
        article.review_report = '{"score": 95, "passed": true}'
        article.quality_score = 95
        article.review_round = 1
        article.error_message = "旧错误"
        article.phase = ArticlePhase.COMPLETED.value
        article.status = ArticleStatus.SUCCESS.value
        article.supervisor_action = SupervisorAction.COMPLETE.value
        session.commit()
    return task_id, model_config


def test_changing_confirmed_title_clears_every_downstream_artifact(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path)
    task_id, original_model_config = _seed_completed_article(workflow)

    changed = workflow.confirm_title(task_id, "新标题")

    assert changed.selected_title == "新标题"
    assert changed.title_options == '[{"title": "原标题"}, {"title": "新标题"}]'
    assert changed.model_config == original_model_config
    assert changed.outline is None
    assert changed.content is None
    assert changed.full_content is None
    assert changed.image_prompts is None
    assert changed.image_results is None
    assert changed.final_markdown is None
    assert changed.review_report is None
    assert changed.quality_score is None
    assert changed.review_round == 0
    assert changed.error_message is None
    assert changed.phase == ArticlePhase.OUTLINE_EDITING.value
    assert changed.status == ArticleStatus.SUCCESS.value
    assert changed.supervisor_action == SupervisorAction.GENERATE_OUTLINE.value


def test_first_title_confirmation_preserves_existing_downstream_artifacts(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path)
    task_id, original_model_config = _seed_completed_article(workflow, selected_title=None)

    confirmed = workflow.confirm_title(task_id, "原标题")

    assert confirmed.outline == '[{"heading": "旧大纲"}]'
    assert confirmed.content == "# 原标题\n\n## 旧正文"
    assert confirmed.full_content == "旧全文"
    assert confirmed.image_prompts == '[{"imageTitle": "旧提示词"}]'
    assert confirmed.image_results == '[{"imageUrl": "/uploads/images/old.png"}]'
    assert confirmed.final_markdown == "# 原标题\n\n旧图文稿"
    assert confirmed.review_report == '{"score": 95, "passed": true}'
    assert confirmed.quality_score == 95
    assert confirmed.review_round == 1
    assert confirmed.error_message == "旧错误"
    assert confirmed.model_config == original_model_config
    assert confirmed.title_options == '[{"title": "原标题"}, {"title": "新标题"}]'


def test_reconfirming_the_same_title_preserves_existing_downstream_artifacts(tmp_path: Path) -> None:
    workflow = _workflow(tmp_path)
    task_id, original_model_config = _seed_completed_article(workflow)

    confirmed = workflow.confirm_title(task_id, "原标题")

    assert confirmed.outline == '[{"heading": "旧大纲"}]'
    assert confirmed.content == "# 原标题\n\n## 旧正文"
    assert confirmed.full_content == "旧全文"
    assert confirmed.image_prompts == '[{"imageTitle": "旧提示词"}]'
    assert confirmed.image_results == '[{"imageUrl": "/uploads/images/old.png"}]'
    assert confirmed.final_markdown == "# 原标题\n\n旧图文稿"
    assert confirmed.review_report == '{"score": 95, "passed": true}'
    assert confirmed.quality_score == 95
    assert confirmed.review_round == 1
    assert confirmed.error_message == "旧错误"
    assert confirmed.model_config == original_model_config
    assert confirmed.title_options == '[{"title": "原标题"}, {"title": "新标题"}]'
    assert confirmed.phase == ArticlePhase.COMPLETED.value
    assert confirmed.status == ArticleStatus.SUCCESS.value
    assert confirmed.supervisor_action == SupervisorAction.COMPLETE.value
