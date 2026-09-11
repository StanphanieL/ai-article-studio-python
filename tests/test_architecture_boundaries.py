"""Executable architecture constraints for the Article Studio application."""

import ast
from collections.abc import Iterable
from pathlib import Path

import pytest

PACKAGE_NAME = "ai_article_studio"
PACKAGE_ROOT = Path(__file__).parents[1] / "src" / PACKAGE_NAME
WEB_ROOT = PACKAGE_ROOT / "web"
_FORBIDDEN_WEB_TARGETS = {
    f"{PACKAGE_NAME}.db",
    f"{PACKAGE_NAME}.domain",
    f"{PACKAGE_NAME}.llm",
    f"{PACKAGE_NAME}.models",
    f"{PACKAGE_NAME}.providers",
}
_FORBIDDEN_DOMAIN_IMPORTS = {"fastapi", "httpx", "pydantic", "sqlalchemy"}
_ROUTE_DECORATORS = {
    "get",
    "post",
    "put",
    "patch",
    "delete",
    "head",
    "options",
    "api_route",
    "websocket",
}


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _dotted_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _resolve_from_base(import_from: ast.ImportFrom, module_name: str) -> str | None:
    if import_from.level == 0:
        return import_from.module

    package_parts = module_name.rsplit(".", maxsplit=1)[0].split(".")
    retained_parts = len(package_parts) - (import_from.level - 1)
    if retained_parts <= 0:
        return None
    base = ".".join(package_parts[:retained_parts])
    return f"{base}.{import_from.module}" if import_from.module else base


def _known_package_modules() -> set[str]:
    """Return importable package modules without hard-coding collaborator names."""

    modules = {PACKAGE_NAME}
    for path in PACKAGE_ROOT.rglob("*.py"):
        relative = path.relative_to(PACKAGE_ROOT.parent).with_suffix("")
        parts = relative.parts[:-1] if path.name == "__init__.py" else relative.parts
        if parts:
            modules.add(".".join(parts))
    return modules


def _is_package_target(target: str) -> bool:
    return target == PACKAGE_NAME or target.startswith(f"{PACKAGE_NAME}.")


def _is_private_workflow_target(target: str) -> bool:
    return any(part.startswith("_workflow_") for part in target.split("."))


def _package_import_targets(tree: ast.Module, module_name: str, known_modules: set[str] | None = None) -> set[str]:
    """Resolve package imports and imported-alias attribute access to full names."""

    known_modules = known_modules or _known_package_modules()
    targets: set[str] = set()
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                if not _is_package_target(imported.name):
                    continue
                targets.add(imported.name)
                aliases[imported.asname or imported.name.split(".", maxsplit=1)[0]] = imported.name
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_from_base(node, module_name)
            if base is None or not _is_package_target(base):
                continue
            targets.add(base)
            for imported in node.names:
                if imported.name == "*":
                    continue
                target = f"{base}.{imported.name}"
                if target in known_modules or _is_private_workflow_target(target):
                    targets.add(target)
                    aliases[imported.asname or imported.name] = target

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        dotted = _dotted_name(node)
        if dotted is None:
            continue
        head, *tail = dotted.split(".")
        if head in aliases:
            targets.add(".".join((aliases[head], *tail)))
    return targets


def _forbidden_web_targets(targets: Iterable[str]) -> set[str]:
    return {
        target
        for target in targets
        if _is_private_workflow_target(target) or any(target == forbidden or target.startswith(f"{forbidden}.") for forbidden in _FORBIDDEN_WEB_TARGETS)
    }


def _route_decorator_names(tree: ast.Module) -> set[str]:
    """Find endpoint decorators by method name, regardless of the receiver."""

    return {
        decorator.func.attr
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute) and decorator.func.attr in _ROUTE_DECORATORS
    }


def test_domain_layer_has_no_framework_or_project_dependencies() -> None:
    """Domain failures and values must remain reusable outside the web application."""

    tree = _parse(PACKAGE_ROOT / "domain.py")
    imports = {
        node.module.split(".", maxsplit=1)[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) if node.level == 0 and node.module is not None
    }
    imports.update(alias.name.split(".", maxsplit=1)[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)

    assert not imports & _FORBIDDEN_DOMAIN_IMPORTS
    assert not _package_import_targets(tree, f"{PACKAGE_NAME}.domain")


def test_web_layer_cannot_reach_infrastructure_or_private_workflow_modules() -> None:
    """Keep request handling on the public service/presentation boundary."""

    forbidden_imports = {
        path: _forbidden_web_targets(_package_import_targets(_parse(path), f"{PACKAGE_NAME}.web.{path.stem}")) for path in WEB_ROOT.glob("*.py")
    }

    assert {path: imports for path, imports in forbidden_imports.items() if imports} == {}


@pytest.mark.parametrize(
    ("module_name", "source", "expected"),
    [
        (f"{PACKAGE_NAME}.web.fixture", "from ai_article_studio.models import Article", {"ai_article_studio.models"}),
        (f"{PACKAGE_NAME}.web.fixture", "import ai_article_studio.models as models_alias", {"ai_article_studio.models"}),
        (f"{PACKAGE_NAME}.fixture", "from .models import Article", {"ai_article_studio.models"}),
        (f"{PACKAGE_NAME}.web.fixture", "from ..models import Article", {"ai_article_studio.models"}),
        (f"{PACKAGE_NAME}.web.fixture", "from ai_article_studio import models as models_alias", {"ai_article_studio.models"}),
        (
            f"{PACKAGE_NAME}.web.fixture",
            "import ai_article_studio as studio\nstudio.providers.create_image_service()",
            {"ai_article_studio.providers.create_image_service"},
        ),
        (f"{PACKAGE_NAME}.web.fixture", "import ai_article_studio._workflow_future as workflow", {"ai_article_studio._workflow_future"}),
        (f"{PACKAGE_NAME}.web.fixture", "from ai_article_studio import _workflow_future as workflow", {"ai_article_studio._workflow_future"}),
        (f"{PACKAGE_NAME}.fixture", "from . import _workflow_future as workflow", {"ai_article_studio._workflow_future"}),
    ],
)
def test_web_import_guard_rejects_absolute_relative_alias_and_private_prefix_bypasses(module_name: str, source: str, expected: set[str]) -> None:
    """Keep future package imports from bypassing the web dependency boundary."""

    targets = _package_import_targets(ast.parse(source), module_name)

    assert expected <= targets
    assert expected <= _forbidden_web_targets(targets)


def test_web_import_guard_keeps_public_facade_and_schema_imports_available() -> None:
    tree = ast.parse("from ..workflow import ArticleWorkflow\nfrom ..schemas import BaseResponse")

    assert _forbidden_web_targets(_package_import_targets(tree, f"{PACKAGE_NAME}.web.fixture")) == set()


def test_composition_root_only_assembles_and_registers_routes() -> None:
    """Keep concrete HTTP endpoint declarations in the ``web`` package."""

    assert _route_decorator_names(_parse(PACKAGE_ROOT / "main.py")) == set()


@pytest.mark.parametrize(
    "source",
    [
        "@app.get('/article')\ndef create(): pass",
        "@router.get('/article')\ndef create(): pass",
        "@api.v1.post('/article')\nasync def create(): pass",
    ],
)
def test_route_guard_detects_fastapi_route_names_for_any_receiver(source: str) -> None:
    assert _route_decorator_names(ast.parse(source))


def test_source_dataclasses_are_frozen_value_objects() -> None:
    """Reserve dataclasses for immutable non-Pydantic/non-ORM source values."""

    mutable_dataclasses: list[Path] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        tree = _parse(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id == "dataclass":
                    mutable_dataclasses.append(path)
                elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name) and decorator.func.id == "dataclass":
                    is_frozen = any(
                        keyword.arg == "frozen" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True for keyword in decorator.keywords
                    )
                    if not is_frozen:
                        mutable_dataclasses.append(path)

    assert mutable_dataclasses == []
