import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (ROOT / path).read_text()


def _function_returns(module: ast.Module, name: str) -> list[ast.Return]:
    returns: list[ast.Return] = []
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            returns.extend(child for child in ast.walk(node) if isinstance(child, ast.Return))
    if not returns:
        raise AssertionError(f"function {name!r} not found")
    return returns


def _assert_returns_only_literal(
    module: ast.Module,
    name: str,
    expected: object,
) -> None:
    for node in _function_returns(module, name):
        value = node.value
        if not isinstance(value, ast.Constant) or value.value is not expected:
            raise AssertionError(f"{name} must only return {expected!r}")


def test_score_facing_serving_has_no_exact_prompt_logits_cache_symbols() -> None:
    source_by_path = {
        "src/torchinferno/runtime/serving.py": _source("src/torchinferno/runtime/serving.py"),
        "src/torchinferno/openai_server.py": _source("src/torchinferno/openai_server.py"),
    }
    prohibited = (
        "_prompt_logits_cache",
        "_prompt_logits_cache_map",
        "_store_prompt_logits_cache",
        "_restore_exact_prompt_logits",
        "TORCHINFERNO_OPENAI_PROMPT_LOGITS_CACHE",
        "_lookup_exact_reusable_prefix",
        "_sample_reusable_prefix_next_token",
        "_sample_reusable_prefix_next_token_list",
    )

    offenders = [
        f"{path}: {marker}"
        for path, source in source_by_path.items()
        for marker in prohibited
        if marker in source
    ]

    assert offenders == []


def test_generated_prefix_logits_gates_remain_inert() -> None:
    serving = ast.parse(_source("src/torchinferno/runtime/serving.py"))
    openai_server = ast.parse(_source("src/torchinferno/openai_server.py"))

    for function_name in (
        "_prefix_cache_store_logits_enabled",
        "_generated_prefix_cache_enabled",
        "_should_collect_generated_prefix_logits",
        "_needs_generated_prefix_logits",
    ):
        _assert_returns_only_literal(serving, function_name, False)
    _assert_returns_only_literal(openai_server, "_online_generated_prefix_cache_enabled", None)
