# AI Article Studio：学习、吃透与面试教程

这不是一份“把每个文件从头读到尾”的说明书。目标是让你按正确顺序理解这个项目，最终能做到四件事：

1. 独立跑通一次 Mock 和真实模型流程；
2. 解释每一步为什么这样设计；
3. 能根据报错定位到对应模块；
4. 在面试中真实、清晰地讲出你的工程贡献与边界。

> 推荐节奏：先用 2 天跑通与画图，再用 5–7 天按模块阅读、修改和测试。不要一开始就阅读 1,700 多行的 `workflow.py`。

### 推荐的真实代码阅读顺序

```text
schemas.py
→ supervisor.py
→ models.py / db.py / config.py
→ workflow.py
→ llm.py
→ jobs.py + main.py 的任务与 SSE 路由
→ providers.py / safe_fetch.py / vision.py
→ anchors.py / composition.py / exports.py
→ templates/ + static/stream.js
→ tests/ 反向验证
```

先读“数据长什么样、状态如何推进”，再读“应用如何组装、任务怎样执行”。这样你不会被路由数量和前端细节淹没。

---

## 0. 先建立正确的项目认知

### 它解决的是什么问题？

它不是“调用一次大模型生成文章”的 Demo，而是一个**可控的多阶段内容生产工作流**：

```text
选题
  → 标题候选与人工确认
  → 大纲
  → 正文（可流式）
  → 配图方案
  → 图片生成 / 检索 / 重试
  → 图文合成
  → 质量评审与一次自动修订
  → Markdown / HTML / ZIP 交付
```

核心设计原则只有三条：

- **LLM 负责内容，不负责流程控制**：下一步由确定性状态机决定。
- **每个阶段的产物都持久化**：刷新、失败、重启后可以继续，而不是从头生成。
- **外部服务默认不可信**：图片 URL、供应商响应和密钥都需要验证、脱敏或隔离。

### 面试中的一句话定位

> 我用 FastAPI 构建了一个单用户本地 LLM 图文创作工作台。重点不只是生成内容，而是用持久化状态机、后台任务、SSE、图片供应商适配和质量门，把多阶段生成做成可编辑、可恢复、可交付的流程。

---

## 1. 第一天：先把产品跑通，不读代码

### 1.1 使用 Mock 模式

不要一开始就填真实 Key。默认没有 `.env` 时就是 Mock 模式，它能生成确定性的标题、正文和示例图片，不产生模型费用。

```bash
uv sync
uv run uvicorn ai_article_studio.main:app --reload --port 8124
```

打开 <http://127.0.0.1:8124/article/list>，创建一个题目，例如“如何用状态机让 AI 内容生成更可控”。

### 1.2 必须亲手完成两条路径

| 路径 | 你要观察什么 | 你要能回答什么 |
|---|---|---|
| 逐步创作 | 标题确认、大纲编辑、正文流式生成、配图、评审和导出 | 为什么中间允许用户修改？修改后哪些产物必须失效？ |
| 一键生成 | 排队、真实子步骤、失败状态、重试入口 | 为什么不能把“一键生成”写成一个阻塞的 HTTP 请求？ |

完成后，在纸上画出上面的流程图，并在每个箭头旁写上“输入、产物、失败后怎么办”。这比立刻读代码重要。

### 1.3 第一个验收点

你应能解释：

- 为什么标题确认是一个人工断点；
- 为什么正文编辑后必须重新生成配图方案；
- 为什么图片生成、图文合成、评审都不能假设一定成功；
- Mock 模式为什么是工程能力，而不是“假的功能”。

---

## 2. 技术栈：先掌握什么，后掌握什么

| 优先级 | 技术 | 在项目中的作用 | 你至少要掌握到什么程度 |
|---:|---|---|---|
| 1 | Python 类型、异常、上下文管理器 | 所有业务代码的基础 | 能读 `dataclass`、`Protocol`、`Enum`、`contextmanager` 和类型标注 |
| 1 | FastAPI + Pydantic | 路由、表单/API 校验、响应模型 | 能解释 `create_app()`、依赖注入、请求校验和 `BaseResponse` |
| 1 | SQLAlchemy + SQLite | 文章与日志持久化 | 能解释 Session、ORM 模型、事务提交和为什么要落库状态 |
| 1 | 有限状态机 | 决定下一步，而不是让 LLM 决策 | 能手画 phase / status / action 三者关系 |
| 2 | 并发与后台任务 | 处理长时间模型/图片调用 | 能解释线程池、队列状态、幂等与安全取消边界 |
| 2 | SSE | 正文逐段返回到页面 | 能解释为什么 SSE 比等待完整 JSON 更适合这里 |
| 2 | httpx + Provider Adapter | 调用模型和图片供应商 | 能解释接口适配、超时、重试、错误转换和 Mock 替身 |
| 2 | Jinja2 + 原生 JavaScript | 服务端渲染、局部更新、流式页面 | 能解释为什么此项目不需要 SPA 才能有好交互 |
| 3 | Markdown / HTML / ZIP | 最终内容交付 | 能解释图片锚点插入、HTML 内嵌和 ZIP 元数据 |
| 3 | 安全下载与密钥管理 | 防止 SSRF、密钥泄露、错误文件落盘 | 能说清 allowlist、重定向、Content-Type 和 `SecretStr` |

不需要为了面试而硬背：Redis、Celery、Docker、Kubernetes、RAG、向量数据库、模型微调。它们不在当前项目的真实边界里。

---

## 3. 第一层代码：配置、入口、数据模型

这层先读完，你才知道“应用是怎么被组装起来的”。

### 3.1 `config.py`：所有运行时配置的唯一入口

阅读顺序：`RuntimeMode` → `Settings` → `get_settings()`。

重点：

- `Settings` 继承 Pydantic Settings，从 `.env` 或环境变量读取配置。
- Key 使用 `SecretStr`，避免普通字符串被随手打印到日志里。
- 默认是 `mock`，真实模式才使用 OpenAI-compatible 地址和供应商密钥。
- 业务代码接收 `Settings` 对象，不直接读取 `os.environ`，这让测试可以注入自己的配置。

面试表达：

> 我把密钥和运行参数收敛到 Settings，业务层只依赖注入后的配置对象；因此 Mock、测试库和真实供应商可以切换，而不会把密钥写进数据库或代码。

### 3.2 `main.py`：Web 层是薄层，不写业务规则

先看 `create_app()`，再看页面路由和 `/api/article/*` 路由。

`create_app()` 负责：加载配置、创建数据库 session factory、装配 `ArticleWorkflow`、`JobRunner`、静态文件和模板。路由的职责是解析请求、调用服务、返回页面或 JSON；复杂规则不要塞到路由中。

建议先只追踪这三条路由：

```text
POST /article/create
POST /article/{task_id}/jobs/next
GET  /api/article/real-stream-generate-content/{task_id}
```

### 3.3 `models.py` 与 `db.py`：什么必须持久化？

`Article` 是工作流的事实来源，保存选题、标题、大纲、正文、图片方案/结果、最终稿、评审、状态、错误和模型配置。`AgentLog` 用来记录每个阶段的输入摘要、输出摘要、耗时和状态。

你必须理解这句话：**数据库保存的是可恢复的工作流状态，不只是最终文章。**

练习：打开 SQLite 数据库或在断点中查看一个任务，列出“刷新页面后仍然需要存在”的字段；再列出“绝对不能保存”的东西（真实 API Key、原始密钥响应等）。

---

## 4. 第二层代码：先吃透状态机，再读工作流

### 4.1 `schemas.py`：跨层的数据契约

重点模型：

- `ArticleModelConfig`：文本模型、长度、图片供应商、图片数量策略；
- `ImagePromptOption` / `ImageResultOption`：图片描述、视觉角色、锚点、来源和结果；
- `QualityReviewResult`：评分、阻断问题、建议和修订结果；
- `WorkflowJobStatus`：页面轮询所需的队列/运行/失败信息。

这些是 Pydantic 契约：前端、路由、工作流和外部模型之间不用靠“猜字段名”协作。

### 4.2 `supervisor.py`：本项目最重要的 200 行左右

先记住三个概念：

```text
Phase  = 文章处于哪个业务阶段，例如 OUTLINE_EDITING
Status = 任务是否在执行，例如 QUEUED / RUNNING / FAILED
Action = 下一步具体要做什么，例如 GENERATE_CONTENT
```

`decide_next_action()` 是纯函数：读取已经持久化的文章状态，返回唯一安全的下一步。它不调用模型、不修改数据库。

为什么这样设计？如果让 LLM 返回“下一步应该生成图片”，流程会不可预测；纯函数既可测试，也能让页面、后台任务和 API 使用同一套判断。

先读 `tests/test_supervisor.py`，再回读 `decide_next_action()`。这是最省力的方式。

### 4.3 `workflow.py`：服务层，按“一个操作”而不是文件顺序读

不要从第一行一直读。按下面顺序搜索方法：

1. `create()`、`get()`、`list()`：任务生命周期；
2. `generate_titles()`、`confirm_title()`、`generate_outline()`、`save_outline()`；
3. `generate_content()` 和流式生成逻辑；
4. `generate_image_prompts()`、`generate_image_results()`、单图重试；
5. `compose()`、`review()`、`apply_review_revision()`；
6. `permanently_delete()`、下游产物清理和 Agent 日志。

每读一个方法，都回答四个问题：

```text
它读取了哪些持久化字段？
它调用了哪个外部/内部服务？
成功时写回哪些字段和状态？
失败时用户下一次能做什么？
```

特别关注“下游失效链”：标题变了会清空大纲、正文、图片和评审；正文变了会清空图片与交付稿。这是保证产物一致性的关键。

---

## 5. 第三层代码：异步任务与 SSE

### 5.1 `jobs.py`：为什么长任务不直接在路由中运行？

图片、模型和评审都有网络延迟。`JobRunner` 使用单用户本地线程池串行执行非流式任务，并维护：

- `QUEUED` / `RUNNING` / `FAILED` 等真实状态；
- 同一任务避免重复提交；
- 一键生成从失败阶段继续，而非错误地从头再跑；
- 删除时在供应商调用的安全边界请求取消。

这里要理解的不是“线程池 API”，而是**任务状态和数据库状态不能混为一谈**：一键生成可能已经成功写入正文，但整体任务仍未结束，因为图片与评审还在后面。

推荐阅读：`tests/test_async_jobs.py`、`tests/test_api_workflow.py`。

### 5.2 SSE：正文为什么特殊？

正文生成直接走事件流：服务器不断发送 `progress`、`content`、`done`、`fail`；浏览器端 `static/stream.js` 用 `EventSource` 累积文本并更新预览。完成或失败后仍要把结果和状态写入数据库。

阅读路径：

```text
main.py 的 real-stream-generate-content 路由
→ workflow.py 的 SSE 生成器
→ static/stream.js 的 EventSource 处理
```

面试问题：为什么不使用普通轮询？

> 对于正文这种连续、可渐进显示的结果，SSE 能降低用户等待感；而图片、评审等离散长任务更适合后台任务加状态轮询。两种交互模式应按结果形态选择。

---

## 6. 第四层代码：LLM 适配与提示词约束

### 6.1 `llm.py`：接口适配，而不是把业务写死在某个供应商上

`ArticleAgent` 是协议，`MockArticleAgent` 用于稳定测试，`OpenAICompatibleArticleAgent` 调用真实模型。工作流依赖协议，不依赖 MiniMax、百炼或某一具体模型。

你要读懂的关键点：

- OpenAI-compatible URL 如何拼接；
- 标题、大纲、图片方案、评审为什么要求 JSON；
- JSON 解析失败后为什么走“修复请求”，而不是直接让任务崩溃；
- 正文为什么既有普通生成，也有流式生成；
- 内容契约如何禁止虚构二维码、数据背书、裸 HTML 或不合适的 JSON 代码块。

建议阅读：`tests/test_llm.py`、`tests/test_quality.py`、`tests/test_legacy_quality_reconcile.py`。

### 6.2 真实 Key 的使用顺序

1. 先用 Mock 通过完整流程；
2. 仅配置文本模型，验证标题/大纲/正文；
3. 再单独配置一个图片供应商；
4. 最后才开启视觉质检。

每次只新增一个变量。出现 401、超时或模型输出格式异常时，先看 Agent 日志和供应商错误，不要一次修改多个 Key、模型和 URL。

---

## 7. 第五层代码：图片供应链是本项目最有工程味的部分

### 7.1 先理解“图片方案”不是一句抽象提示词

`ImagePromptOption` 除中英文提示词外，还包含：视觉角色、图片标题、用途、图库检索词、章节锚点、插入位置和展示布局。

因此图片并不是永远插在某段文字后，而是先由模型理解正文结构，再在合成时按锚点插入；锚点变化时会有安全回退。

### 7.2 `providers.py`：统一供应商接口

阅读顺序：

```text
calculate_image_count()
→ ImageGenerationService
→ MockImageProvider
→ PexelsImageProvider / SiliconFlowImageProvider / GoogleAIImageProvider
→ _auto_provider_order()
→ persist_remote_image()
```

要能解释：

- 显式选择供应商失败时为什么应该报错，而不是偷偷换图；
- AUTO 为什么要根据视觉角色区分图库照片和技术图解；
- 为什么生成型图片必须立即下载到本地，而不能只保存短期 URL；
- 单图重试为什么只替换数组指定下标。

### 7.3 `safe_fetch.py`：安全不是“多写几个 if”

外部图片 URL 可能指向错误资源、发生恶意重定向或伪装 Content-Type。该模块检查来源 allowlist、重定向和真实图片内容类型后才允许下载或导出。

推荐阅读：`tests/test_providers.py`、`tests/test_exports.py`、`tests/test_vision.py`。

---

## 8. 第六层代码：图文合成、质量评审与导出

### 8.1 `composition.py` 与 `anchors.py`

这两个模块解决“图应该插在哪里”。先从正文抽取标题和段落锚点，再将封面、章节图或段落图插入 Markdown；若锚点不可用，回退到同章节安全位置。最终仅保留读者需要的来源和作者信息。

### 8.2 `quality.py` 与 `vision.py`

质量门分为两类：

- **文本结构与事实性检查**：Markdown 标题、代码围栏、裸 HTML、虚构资源和不当承诺；
- **可选视觉检查**：当显式配置视觉模型时，检查图片与提示词的相关性和明显瑕疵。

阻断问题不能被高分掩盖；自动修订最多一轮，之后转为人工处理。这是“质量循环可控”的边界，而不是无限调用模型。

### 8.3 `exports.py`

最终交付包括：

- Markdown 原稿；
- HTML 预览，图片可内嵌；
- ZIP，其中包含 `article.md`、`article.html`、`metadata.json` 和本地图片。

面试时可以强调：导出不是简单下载字符串，还要处理图片去重、来源元数据和安全 URL。

---

## 9. 前端：只掌握“界面如何和状态机协作”

前端不是你的主要学习瓶颈。请按下面顺序理解：

1. `templates/list.html`：创建任务、模型设置、任务卡片；
2. `templates/detail.html`：五步卡片向导和进度条；
3. `_workflow_panel.html`：各阶段具体内容；
4. `static/stream.js`：SSE、表单异步提交、任务轮询、失败恢复；
5. `static/app.css` 与 `client-ui.css`：视觉样式，不要先花大量时间逐行读。

你应该能讲清：Jinja2 负责首次服务端渲染，原生 JavaScript 的 `fetch` 负责局部更新和表单提交，`EventSource` 负责正文 SSE，后台任务状态通过轮询刷新。这里选择轻量服务端渲染，是为了让 Python 后端主导业务，而不是为了实现复杂的前端状态管理。

---

## 10. 推荐的 8 天学习计划

| 天数 | 目标 | 必须完成的动作 | 当天产出 |
|---:|---|---|---|
| Day 1 | 跑通产品 | Mock 模式完成逐步与一键流程 | 一张流程图 + 5 个产品观察 |
| Day 2 | 吃透流程 | 读 `schemas.py`、`supervisor.py`、对应测试 | phase/status/action 对照表 |
| Day 3 | 建立骨架 | 读 `models.py`、`db.py`、`config.py` 和 `main.py` 的 `create_app()` | 应用装配图 + Article 字段说明 |
| Day 4 | 理解业务 | 按操作阅读 `workflow.py` | 一张“产物失效链”图 |
| Day 5 | 理解模型 | 读 `llm.py`、`quality.py` | 一份 JSON 修复和质量门笔记 |
| Day 6 | 理解长任务 | 读 `jobs.py`、SSE 路由和 `stream.js` | 解释同步、后台任务、SSE 的选择 |
| Day 7 | 理解图片 | 读 `providers.py`、`safe_fetch.py`、`composition.py` | 一张图片从提示词到导出的数据流图 |
| Day 8 | 面试演练 | 运行测试，选择一个小修改并补测试 | 1 分钟讲稿 + 3 个深挖回答 |

如果时间只有 3 天，优先完成 Day 1、Day 3、Day 5、Day 7 和 Day 8。

---

## 11. 建议亲手完成的 6 个小练习

先建分支，再做以下练习；每个练习都要先写或修改测试。

1. **状态机练习**：给 `SupervisorAction` 增加一个仅用于展示的动作标签，理解枚举、路由与页面提示如何关联。
2. **配置练习**：新增一个不敏感的展示配置，例如默认文章风格，并让它从 Settings → 页面表单 → `ArticleModelConfig` 完整流动。
3. **失败恢复练习**：用 Mock Agent 故意让图片方案返回不合法 JSON，观察 JSON 修复、失败持久化与重试状态。
4. **供应商练习**：阅读 `MockImageProvider`，新增一个只在测试使用的假供应商，验证显式选择失败不会被 AUTO 偷偷替换。
5. **质量门练习**：为“未闭合代码围栏”或“裸 HTML”新增一条测试，理解阻断问题如何阻止交付。
6. **导出练习**：为 ZIP `metadata.json` 新增一个非敏感字段，并补充导出测试。

练习原则：一次只改一层；先跑对应测试，再跑全量 `uv run pytest`。不要直接用真实 API 调试逻辑问题。

---

## 12. 面试高频问题与答题骨架

### 为什么不用一个 Prompt 一次性生成全部内容？

因为标题、大纲、正文、图片和评审的失败方式不同，用户也需要在关键节点修改内容。拆成持久化阶段后，可以局部重试、追踪错误并避免不必要的重复调用。

### 为什么 Supervisor 是确定性的，而不是再让一个 LLM 决定下一步？

下一步取决于已有产物和任务状态，属于业务控制逻辑。确定性函数可测试、可重放，也避免模型格式异常导致流程跳错阶段；LLM 只负责生成和评审内容。

### 逐步生成、一键生成和 SSE 的关系是什么？

逐步/一键是工作流模式；图片、评审等离散长操作进入后台任务并轮询状态；正文需要逐段可见，因此使用 SSE。三者解决的是不同层次的问题。

### 修改正文后，为什么要清理图片和评审？

图片锚点、图片提示词和质量结论都依赖正文。保留它们会造成“正文已经变了，但配图和评审还是旧版本”的一致性错误。

### 为什么同时有 Mock 和真实 Provider？

Mock 让工作流、页面和测试不依赖费用、网络和第三方稳定性；真实 Provider 通过适配器接入。这样可以把“业务逻辑正确”与“外部服务可用”分开验证。

### 你如何避免图片下载的安全问题？

只允许本地文件或已知供应商域名，验证重定向和内容类型，生成图立即落本地；导出时再次使用安全下载边界，不接受任意用户 URL。

### 项目的限制是什么？

它是单用户本地应用，未实现登录、多租户、分布式任务队列、生产级密钥管理或高并发扩展。图片和模型质量仍受供应商能力与提示词影响。

---

## 13. 最终自检清单

在把项目写进简历前，确保你能不看代码回答：

- [ ] 说清产品流程和用户在哪三处拥有编辑权；
- [ ] 手画 `Phase / Status / Action` 的区别；
- [ ] 解释一键生成为什么可从失败阶段恢复；
- [ ] 解释正文 SSE 与后台图片任务为什么不同；
- [ ] 说清一次正文修改会失效哪些产物；
- [ ] 讲清 AUTO 图片路由、显式供应商和单图重试；
- [ ] 讲清 API Key、图片 URL 和导出的安全边界；
- [ ] 能在 1 分钟内讲完项目，并在 5 分钟内展开状态机或图片供应链；
- [ ] 独立运行 `uv run pytest` 并知道主要测试文件分别覆盖什么。

当以上每一项都能回答时，你不只是“用过 AI Coding 做了这个项目”，而是能够解释、验证和继续迭代这套 Python 工程系统的人。
