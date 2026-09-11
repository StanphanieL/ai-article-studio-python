# Article Studio architecture and style guide

AI Article Studio borrows K9Forge's engineering discipline, not its terminal product shape. It is a local, single-user FastAPI workbench whose primary contract is a browser-based, persisted article workflow. The application keeps that contract while making ownership and dependencies visible.

## Ownership map

| Area | Owns | Must not own |
| --- | --- | --- |
| `main.py` | application assembly, lifespan, static mounts, exception mapping, router registration | page/API handlers or workflow policy |
| `web/` | HTTP routes, request dependencies, page contexts, response serialization | database sessions, adapter construction, SQLAlchemy models, or private workflow collaborators |
| `workflow.py` | public workflow facade, shared workflow state, locks, jobs/SSE coordination, and compatibility methods | HTTP request/response details |
| `_workflow_*.py` | one cohesive private workflow concern, through the facade's shared context | independent sessions, adapters, locks, or a second state machine |
| `contracts.py` | typed adapter ports | vendor HTTP or persistence details |
| `domain.py` | dependency-free errors and immutable domain values | FastAPI, Pydantic, SQLAlchemy, or provider code |
| `llm.py` / `providers.py` | concrete vendor adaptation and safe error normalization | route handling or workflow transitions |
| `models.py` / `db.py` | SQLAlchemy persistence and session construction | web presentation or provider calls |
| `schemas.py` | Pydantic transport/configuration validation models | workflow orchestration or I/O |

`tests/test_architecture_boundaries.py` makes the most important parts of this map executable: routes cannot migrate back into the composition root; the web package cannot reach persistence, adapters, domain errors, or private workflow collaborators directly; and source dataclasses must be frozen. The test is a guardrail, not a substitute for reviewing ownership when a new feature crosses layers.

## Style guide

1. Give every behavior one explicit owner. Put request handling in `web/`, state transitions in the deterministic supervisor/workflow, vendor I/O in adapters, and persistence in models/session helpers.
2. Use typed interfaces at boundaries. Protocols in `contracts.py` describe adapter capabilities; Pydantic models validate request, configuration, and persisted JSON shapes; route dependencies expose concrete application services deliberately.
3. Keep dependencies narrow and directional. Web code calls the public workflow facade and presentation helpers; it never constructs adapters or reaches database/ORM/private-collaborator internals. Domain values depend only on the standard library.
4. Use a dataclass only for an immutable, dependency-light value that is not a Pydantic request/response/configuration model or a SQLAlchemy record. Mark it `frozen=True`; use an ordinary private class for short-lived mutable parser state.
5. Reorganize behind compatibility facades. Public imports and established `ArticleWorkflow` methods remain stable while private collaborators move; do not force callers to follow an internal extraction.
6. Prefer small, reviewable changes. Run Ruff for source and tests, preserve public endpoint/payload behavior, and add a boundary test whenever a new layer or dependency direction is introduced.

### Current boundary phase

The text-agent seam is a Protocol today. Image generation and visual review are
injected into `ArticleWorkflow`, but remain concrete service seams while their
contracts settle. This guide therefore does not claim that every dependency is
already port-driven. When either service needs to vary independently, introduce
a small service-level Protocol in `contracts.py` and then tighten the workflow
import boundary; do not add a second owner in the meantime.

## Why this is not a K9Forge CLI/Textual clone

K9Forge's CLI/Textual layout is designed for an interactive terminal product. Article Studio instead needs browser forms, server-rendered pages, SSE article streaming, image preview/retry controls, downloadable files, and workflow state that survives each HTTP request. FastAPI, Jinja2, vanilla browser JavaScript, and synchronous SQLAlchemy are therefore intentional product boundaries—not style debt. Copying a terminal UI or adding Textual/MCP dependencies would replace those delivery contracts without improving the current workflow.

The shared standard is architectural: explicit ownership, typed contracts, narrow dependencies, immutable value objects, and safe compatibility facades.
