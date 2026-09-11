"""Request-scoped access to application services assembled by ``main``."""

from typing import Annotated

from fastapi import Depends, Request
from fastapi.templating import Jinja2Templates

from ..config import Settings
from ..jobs import JobRunner
from ..workflow import ArticleWorkflow


def get_workflow(request: Request) -> ArticleWorkflow:
    return request.app.state.workflow


def get_job_runner(request: Request) -> JobRunner:
    return request.app.state.job_runner


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_templates(request: Request) -> Jinja2Templates:
    return request.app.state.templates


WorkflowDep = Annotated[ArticleWorkflow, Depends(get_workflow)]
JobRunnerDep = Annotated[JobRunner, Depends(get_job_runner)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
TemplatesDep = Annotated[Jinja2Templates, Depends(get_templates)]
