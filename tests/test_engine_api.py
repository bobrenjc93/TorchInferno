from __future__ import annotations

import asyncio

from torchinferno.engine import (
    AsyncInferenceEngine,
    CacheConfig,
    EngineConfig,
    GenerateRequest,
    InferenceEngine,
    ModelConfig,
    SamplingConfig,
    SchedulerConfig,
)
from torchinferno.server import (
    chat_completion_response,
    model_list_response,
    parse_chat_completion_request,
)


def _tiny_engine_config() -> EngineConfig:
    return EngineConfig(
        model="tiny",
        model_kind="tiny-deepseek",
        tokenizer="byte",
        device="cpu",
        dtype="float32",
        max_model_len=32,
        max_batch_size=1,
        batch_wait_ms=0.0,
    )


def test_engine_config_groups_model_cache_and_scheduler_options() -> None:
    config = EngineConfig.from_parts(
        ModelConfig(model="tiny", model_kind="tiny-deepseek", tokenizer="byte", device="cpu"),
        cache=CacheConfig(backend="paged", page_size=8),
        scheduler=SchedulerConfig(max_batch_size=2, batch_wait_ms=3.0),
    )

    legacy = config.to_legacy_openai_config(host="127.0.0.1", port=8123)

    assert config.cache_backend == "paged"
    assert config.page_size == 8
    assert config.max_batch_size == 2
    assert legacy.host == "127.0.0.1"
    assert legacy.port == 8123
    assert legacy.model_kind == "tiny-deepseek"


def test_inference_engine_generates_from_raw_prompt() -> None:
    engine = InferenceEngine.from_config(_tiny_engine_config())
    try:
        output = engine.generate(
            GenerateRequest(
                request_id="prompt-1",
                prompt="hi",
                sampling=SamplingConfig(max_tokens=2, temperature=0.0),
            )
        )
    finally:
        engine.close()

    assert output.request_id == "prompt-1"
    assert len(output.token_ids) == 2
    assert output.usage.prompt_tokens == 2
    assert output.usage.completion_tokens == 2
    assert output.usage.total_tokens == 4


def test_inference_engine_generates_from_chat_messages() -> None:
    engine = InferenceEngine.from_config(_tiny_engine_config())
    try:
        output = engine.generate(
            GenerateRequest(
                request_id="chat-1",
                messages=[{"role": "user", "content": "Say hi"}],
                sampling=SamplingConfig(max_tokens=1, temperature=0.0),
            )
        )
    finally:
        engine.close()

    assert output.request_id == "chat-1"
    assert len(output.token_ids) == 1
    assert output.usage.prompt_tokens > 0
    assert output.usage.completion_tokens == 1


def test_async_inference_engine_streams_tokens() -> None:
    async def run() -> list[object]:
        engine = AsyncInferenceEngine.from_config(_tiny_engine_config())
        try:
            events = []
            async for event in engine.generate_stream(
                GenerateRequest(
                    request_id="stream-1",
                    prompt="go",
                    sampling=SamplingConfig(max_tokens=2, temperature=0.0),
                    stream=True,
                )
            ):
                events.append(event)
            return events
        finally:
            await engine.aclose()

    events = asyncio.run(run())

    assert [event.token_id for event in events[:-1]]
    assert events[-1].finished is True
    assert events[-1].usage is not None
    assert events[-1].usage.completion_tokens == 2


def test_openai_protocol_helpers_are_engine_neutral() -> None:
    request = parse_chat_completion_request(
        {
            "model": "tiny",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 3,
            "temperature": 0.0,
            "stream": True,
        }
    )
    models = model_list_response("tiny", created=1)
    response = chat_completion_response(
        model_id="tiny",
        content="ok",
        prompt_tokens=4,
        completion_tokens=2,
        created=1,
        completion_id="chatcmpl-test",
    )

    assert request.stream is True
    assert request.max_tokens == 3
    assert models["data"][0]["id"] == "tiny"  # type: ignore[index]
    assert response["usage"]["total_tokens"] == 6  # type: ignore[index]
