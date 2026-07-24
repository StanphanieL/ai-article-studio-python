# AI Article Studio

[English](README.md) · [中文](README.zh-CN.md)

> 从一个选题，到可交付的图文稿。

![AI Article Studio 首页](docs/assets/home.png)

AI Article Studio 是一个单用户本地创作工作台：你可以逐步确认标题、大纲、正文与配图，也可以一键完成整篇图文稿。它将可编辑的创作流程、图像供应链、质量评审和交付导出组合在一个 Python 应用中。

## 核心能力

- **逐步创作与一键完成**：后台任务会展示真实阶段、失败原因和可重试入口。
- **流式正文生成**：通过 SSE 实时呈现正文，并保留 Markdown 编辑与预览。
- **智能配图**：支持 Mock、Pexels、SiliconFlow、Google AI 与按视觉角色自动分流；可调整位置、单图重试并记录来源。
- **质量与交付**：内容质量闸门、最多一次自动修订，以及 Markdown、HTML、ZIP 三种导出。
- **本地优先与安全边界**：密钥仅从本地环境变量读取；下载与导出仅接受受信任来源；默认不包含登录或多用户能力。

## 技术栈

Python 3.13 · uv · FastAPI · SQLAlchemy · SQLite / PostgreSQL · Pydantic · Jinja2 · HTMX · SSE · httpx · pytest · Ruff

## 快速启动

```bash
uv sync
uv run uvicorn ai_article_studio.main:app --reload --port 8124
```

打开 <http://127.0.0.1:8124/article/list>。未创建 `.env` 时，应用默认以 Mock 模式运行，可完整体验工作流而不会调用外部模型。

## 接入真实模型

复制 `.env.example` 为 `.env`，填入 OpenAI-compatible 文本模型及需要的图片供应商密钥。百炼（DashScope）可作为 OpenAI-compatible 端点使用；Pexels、SiliconFlow 和 Google AI 均可按需启用。

```bash
cp .env.example .env
```

## 验证

```bash
uv run pytest
uv run ruff check .
node --check src/ai_article_studio/static/stream.js
```

## 项目边界

这是一个单用户本地演示应用，不包含登录、多租户数据隔离或托管式密钥管理。
