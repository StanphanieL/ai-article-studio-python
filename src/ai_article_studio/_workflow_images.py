"""Private image generation checkpoints, provider compatibility, and visual review.

All services and execution context are read from the facade at call time so
configured adapters and test doubles retain their existing injection boundary.
"""

from __future__ import annotations

import inspect
import json
import time
from typing import TYPE_CHECKING

from ._workflow_persistence import safe_error_message as _safe_error_message
from .domain import WorkflowCancellationRequested, WorkflowValidationError
from .models import Article
from .providers import calculate_image_count
from .schemas import ImageCountMode, ImagePromptOption, ImageProvider, ImageResultOption, ImageVisualReview, ImageVisualReviewStatus
from .supervisor import ArticlePhase, ArticleStatus, SupervisorAction

if TYPE_CHECKING:
    from .workflow import ArticleWorkflow


class _WorkflowImages:
    def __init__(self, workflow: ArticleWorkflow) -> None:
        self._workflow = workflow

    def generate_image_prompts(self, article: Article) -> str:
        workflow = self._workflow
        if not article.content or not article.selected_title:
            raise WorkflowValidationError("请先生成正文")
        config = workflow._config(article)
        count = config.image_count if config.image_count_mode is ImageCountMode.CUSTOM else calculate_image_count(article.content)
        prompts = workflow._agent_for(article).image_prompts(article.topic, article.selected_title, article.content, article.style, count or 3)
        prompts, adjusted = workflow._normalize_generated_image_plan(article.content, prompts)
        workflow._validate_image_plan_contract(article.content, prompts)
        article.image_prompts = json.dumps([item.model_dump(by_alias=True) for item in prompts], ensure_ascii=False)
        article.image_results = article.final_markdown = None
        article.review_report = article.quality_score = None
        article.phase = ArticlePhase.CONTENT_GENERATION.value
        article.status = ArticleStatus.SUCCESS.value
        return "配图方案已生成" + (f"；已自动校准 {adjusted} 项图片编排" if adjusted else "")

    def generate_images_incrementally(self, task_id: str, requested_provider: ImageProvider) -> None:
        """Generate and persist each asset at an independent recovery boundary.

        A provider can fail on image 3 after two successful, billable calls.
        Keeping the earlier assets prevents a retry from repeating calls or
        making the user wait through the whole batch again.
        """
        workflow = self._workflow

        with workflow.task_write_lock(task_id):
            article = workflow.get(task_id)
            workflow._ensure_article_writable(article)
            raw_prompts = workflow._prompts(article)
            prompts = [workflow._with_anchor_excerpt(article.content or "", prompt) for prompt in raw_prompts]
            if not prompts:
                raise WorkflowValidationError("请先生成图片提示词")
            if prompts != raw_prompts:
                # Existing task records predate anchor excerpts. Upgrade them
                # lazily at the safe pre-generation boundary so retrying an
                # older plan gets the same grounded provider prompt as a newly
                # created plan, without forcing the user to regenerate it.
                workflow._update(task_id, image_prompts=workflow._dump_image_records(prompts))
            results = workflow._images(article)
            if len(results) > len(prompts):
                raise WorkflowValidationError("图片结果数量异常，请重新生成图片素材")

            config = workflow._config(article)
            if requested_provider is ImageProvider.AUTO and config.image_provider is not ImageProvider.AUTO:
                # An article-level setting is an explicit user choice in both
                # manual and one-click flows. Only AUTO is allowed to try a
                # different supplier or a local fallback.
                provider = config.image_provider
            else:
                provider = requested_provider

            for index in range(len(results), len(prompts)):
                if workflow.cancellation_requested(task_id):
                    raise WorkflowCancellationRequested("任务已请求取消")
                workflow._record_image_generation_progress(task_id, results, index, len(prompts), provider)
                started = time.perf_counter()
                try:
                    result = workflow._generate_image_for_task(
                        prompts[index],
                        provider,
                        index,
                        task_id,
                    )
                except Exception as error:
                    workflow._record_image_generation_failure(task_id, index, len(prompts), provider, error, started)
                    raise
                results.append(result)
                workflow._record_image_generation_progress(task_id, results, index + 1, len(prompts), provider, started=started)

    def record_image_generation_progress(
        self,
        task_id: str,
        results: list[ImageResultOption],
        completed: int,
        total: int,
        provider: ImageProvider,
        *,
        started: float | None = None,
    ) -> None:
        """Publish real progress and keep the successfully generated prefix."""
        workflow = self._workflow

        with workflow.sessions() as session:
            article = workflow._get_article(session, task_id)
            article.image_results = workflow._dump_image_records(results) if results else None
            article.final_markdown = None
            article.review_report = article.quality_score = None
            article.phase = ArticlePhase.CONTENT_GENERATION.value
            article.supervisor_action = workflow.encode_image_action(SupervisorAction.GENERATE_IMAGE_RESULTS.value, provider)
            article.supervisor_reason = f"已生成 {completed}/{total} 张图片" if completed >= total else f"正在生成第 {completed + 1}/{total} 张图片"
            if workflow._background_job_execution.get():
                article.status = ArticleStatus.RUNNING.value
            else:
                article.status = ArticleStatus.SUCCESS.value
            article.error_message = None
            article.update_time = workflow._clock()
            if started is not None:
                workflow._log(
                    session,
                    task_id,
                    "ImageResultAgent",
                    f"image={completed}/{total}; provider={provider.value}",
                    article.supervisor_reason,
                    started,
                )
            session.commit()

    def record_image_generation_failure(
        self,
        task_id: str,
        index: int,
        total: int,
        provider: ImageProvider,
        error: Exception,
        started: float,
    ) -> None:
        workflow = self._workflow
        with workflow.sessions() as session:
            article = workflow._get_article(session, task_id)
            article.status = ArticleStatus.FAILED.value
            article.supervisor_action = workflow.encode_image_action(SupervisorAction.GENERATE_IMAGE_RESULTS.value, provider)
            article.supervisor_reason = f"第 {index + 1}/{total} 张图片生成失败，可重试剩余图片"
            article.error_message = _safe_error_message(error)
            article.update_time = workflow._clock()
            workflow._log(
                session,
                task_id,
                "ImageResultAgent",
                f"image={index + 1}/{total}; provider={provider.value}",
                None,
                started,
                error,
            )
            session.commit()

    def regenerate_image_result(self, article: Article, index: int, provider: ImageProvider) -> str:
        workflow = self._workflow
        raw_prompts, results = workflow._prompts(article), workflow._images(article)
        prompts = [workflow._with_anchor_excerpt(article.content or "", prompt) for prompt in raw_prompts]
        if prompts != raw_prompts:
            article.image_prompts = workflow._dump_image_records(prompts)
        if index < 0 or index >= len(prompts) or len(results) > len(prompts):
            raise WorkflowValidationError("图片索引或图片数据不合法")
        if index > len(results):
            raise WorkflowValidationError("请先重试前一张未完成的图片")
        generated = workflow._generate_image_for_task(prompts[index], provider, index, article.task_id)
        if index == len(results):
            results.append(generated)
        else:
            results[index] = generated
        article.image_results = json.dumps([item.model_dump(by_alias=True) for item in results], ensure_ascii=False)
        article.final_markdown = None
        article.review_report = article.quality_score = None
        article.phase = ArticlePhase.CONTENT_GENERATION.value
        article.status = ArticleStatus.SUCCESS.value
        article.supervisor_action = None
        article.supervisor_reason = "单张图片已重试，请继续合成图文稿并重新评审"
        return article.image_results

    def review_generated_images(self, article: Article, indices: list[int] | None = None) -> str:
        """Attach optional visual checks without changing workflow state or assets."""
        workflow = self._workflow

        prompts, results = workflow._prompts(article), workflow._images(article)
        if len(prompts) != len(results):
            raise WorkflowValidationError("图片结果与配图方案数量不一致，无法执行视觉质检")
        selected_indices = indices if indices is not None else list(range(len(results)))
        for index in selected_indices:
            if index < 0 or index >= len(results):
                raise WorkflowValidationError("图片索引不存在，无法执行视觉质检")
            results[index] = results[index].model_copy(update={"visual_review": workflow._safe_visual_review(results[index], prompts[index])})
        article.image_results = workflow._dump_image_records(results)
        counts: dict[str, int] = {}
        for index in selected_indices:
            status = results[index].visual_review.status.value
            counts[status] = counts.get(status, 0) + 1
        return "视觉质检：" + "，".join(f"{status} {count}" for status, count in sorted(counts.items()))

    def safe_visual_review(self, image: ImageResultOption, prompt: ImagePromptOption):
        workflow = self._workflow
        try:
            return workflow.vision_reviewer.review(image, prompt)
        except Exception:
            # Third-party visual review must never fail a usable generated image.
            return ImageVisualReview(
                status=ImageVisualReviewStatus.UNAVAILABLE,
                issues=["视觉质检结果不可用，可按需重试。"],
            )

    def generate_images_for_task(
        self,
        prompts: list[ImagePromptOption],
        provider: ImageProvider,
        task_id: str,
    ) -> list[ImageResultOption]:
        workflow = self._workflow
        generate = workflow.images.generate
        if accepts_task_id(generate):
            return generate(prompts, provider, task_id=task_id)
        return generate(prompts, provider)

    def generate_image_for_task(
        self,
        prompt: ImagePromptOption,
        provider: ImageProvider,
        index: int,
        task_id: str,
    ) -> ImageResultOption:
        workflow = self._workflow
        generate_one = workflow.images.generate_one
        kwargs: dict[str, object] = {}
        if accepts_task_id(generate_one):
            kwargs["task_id"] = task_id
        return generate_one(prompt, provider, index, **kwargs)


def accepts_task_id(operation: object) -> bool:
    try:
        parameters = inspect.signature(operation).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(parameter.name == "task_id" or parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)
