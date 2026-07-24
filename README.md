# AI Article Studio

[English](README.md) · [中文](README.zh-CN.md)

> From one topic to a reviewed, export-ready illustrated article.

![AI Article Studio homepage](docs/assets/home.png)

AI Article Studio is a single-user local workspace for producing illustrated articles. Review a topic, title, outline, article, and visuals step by step—or let the workflow complete the entire draft in one run.

## Highlights

- **Guided and one-click workflows** with background jobs, truthful progress, retry points, and recovery after interruption.
- **SSE article streaming** with Markdown editing and preview.
- **Visual planning and generation** with Mock, Pexels, SiliconFlow, Google AI, and visual-role-aware automatic routing; placement is editable and each image can be retried independently.
- **Quality and delivery** through content gates, one automatic revision round, and Markdown, HTML, or ZIP exports.
- **Local-first credentials**: API keys stay in environment variables and are never committed, logged, or returned by the application.

## Tech Stack

Python 3.13 · uv · FastAPI · SQLAlchemy · SQLite / PostgreSQL · Pydantic · Jinja2 · HTMX · SSE · httpx · pytest · Ruff

## Run Locally

```bash
uv sync
uv run uvicorn ai_article_studio.main:app --reload --port 8124
```

Visit <http://127.0.0.1:8124/article/list>. Without a local `.env`, the app runs in deterministic Mock mode and does not contact external AI providers.

## Connect Real Providers

Copy `.env.example` to `.env`, then configure an OpenAI-compatible text model and only the image providers you need. DashScope can be used as an OpenAI-compatible endpoint; Pexels, SiliconFlow, and Google AI are optional.

```bash
cp .env.example .env
```

## Verify

```bash
uv run pytest
uv run ruff check .
node --check src/ai_article_studio/static/stream.js
```

## Project Boundary

This is a local, single-user demonstration application. It intentionally does not include authentication, multi-tenant data isolation, or hosted secret management.
