"""Executable architecture constraints for the Article Studio application."""

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[1] / "src" / "ai_article_studio"
WEB_ROOT = PACKAGE_ROOT / "web"
_FORBIDDEN_WEB_MODULES = {
    "db",
    "domain",
    "llm",
    "models",
    "providers",
    "_workflow_execution",
    "_workflow_image_plan",
    "_workflow_images",
    "_workflow_persistence",
}
_FORBIDDEN_DOMAIN_IMPORTS = {"fastapi", "httpx", "pydantic", "sqlalchemy"}
_ROUTE_DECORATORS = {
    "get",
    "post",
    "put",
    "patch",
    "delete",
    "api_route",
    "websocket",
}


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _relative_import_root(import_from: ast.ImportFrom) -> str | None:
    if import_from.level != 2 or import_from.module is None:
        return None
    return import_from.module.split(".", maxsplit=1)[0]


def test_domain_layer_has_no_framework_or_project_dependencies() -> None:
    """Domain failures and values must remain reusable outside the web application."""

    tree = _parse(PACKAGE_ROOT / "domain.py")
    imports = {
        node.module.split(".", maxsplit=1)[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) if node.level == 0 and node.module is not None
    }
    imports.update(alias.name.split(".", maxsplit=1)[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)

    assert not imports & _FORBIDDEN_DOMAIN_IMPORTS
    assert not any(isinstance(node, ast.ImportFrom) and node.level > 0 for node in ast.walk(tree))


def test_web_layer_cannot_reach_infrastructure_or_private_workflow_modules() -> None:
    """Keep request handling on the public service/presentation boundary."""

    forbidden_imports: dict[Path, set[str]] = {}
    for path in WEB_ROOT.glob("*.py"):
        imports = {
            root for node in ast.walk(_parse(path)) if isinstance(node, ast.ImportFrom) if (root := _relative_import_root(node)) in _FORBIDDEN_WEB_MODULES
        }
        if imports:
            forbidden_imports[path] = imports

    assert forbidden_imports == {}


def test_composition_root_only_assembles_and_registers_routes() -> None:
    """Keep concrete HTTP endpoint declarations in the ``web`` package."""

    route_decorators = [
        decorator
        for node in ast.walk(_parse(PACKAGE_ROOT / "main.py"))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Call)
        if isinstance(decorator.func, ast.Attribute)
        if isinstance(decorator.func.value, ast.Name)
        if decorator.func.value.id == "app"
        if decorator.func.attr in _ROUTE_DECORATORS
    ]

    assert route_decorators == []


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
