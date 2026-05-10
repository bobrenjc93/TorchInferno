from __future__ import annotations

from torchinferno.cli import _microbench_messages, build_parser as build_cli_parser
from torchinferno.openai_server import config_from_args as openai_config_from_args


def test_main_cli_openai_server_parser_matches_openai_server_config() -> None:
    parser = build_cli_parser()

    args = parser.parse_args(
        [
            "openai-server",
            "--model",
            "tiny",
            "--model-kind",
            "tiny-deepseek",
            "--tokenizer",
            "byte",
            "--device",
            "cpu",
            "--single-request-admission-wait-ms",
            "0",
        ]
    )
    config = openai_config_from_args(args)

    assert config.single_request_admission_wait_ms == 0.0
    assert config.model_kind == "tiny-deepseek"


def test_openai_microbench_self_consistency_messages_are_identical() -> None:
    first = _microbench_messages(3, 0, "self-consistency")
    second = _microbench_messages(3, 7, "self-consistency")

    assert first == second
    assert first[0]["role"] == "system"
    assert first[1]["content"] == "17 * 23 ="
