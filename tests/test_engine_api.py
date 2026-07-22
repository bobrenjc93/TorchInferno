from __future__ import annotations

import asyncio
import os
import socket

import pytest
import torch

from torchinferno.engine import loader as loader_module
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
from torchinferno.engine.loader import (
    initialize_standard_tensor_parallel_runtime,
    load_model_for_engine,
)


def _standard_tp_runtime_worker(
    rank: int,
    world_size: int,
    port: int,
    results: object,
) -> None:
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(rank),
        }
    )
    torch.cuda.is_available = lambda: False  # type: ignore[method-assign]
    try:
        device = initialize_standard_tensor_parallel_runtime(world_size)
        import torchinferno.models.deepseek_v4.tensor_parallel as v4_tp
        import torchinferno.runtime.sampling as sampling

        v4_tp.world_size = world_size
        v4_tp.rank = rank
        v4_tp.set_tensor_parallel_process_group(None)
        sampling.sample_next_token = lambda logits, temperature: torch.full(  # type: ignore[assignment]
            (logits.size(0),),
            rank + 7,
            dtype=torch.long,
            device=logits.device,
        )
        fake_model = type(
            "FakeV4Sampler",
            (),
            {
                "args": type("Args", (), {"vocab_size": 4})(),
                "head": type("Head", (), {"part_vocab_size": 2})(),
                "tensor_parallel_rank": rank,
            },
        )()
        token = v4_tp.DeepSeekV4TensorParallelForCausalLM._sample_next_token(
            fake_model,
            torch.tensor([[float(rank), float(rank + 1)]]),
            0.7,
        )
        results.put(
            (
                rank,
                str(device),
                torch.distributed.get_world_size(),
                int(token.item()),
            )
        )
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _unused_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


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


def test_disaggregation_rejects_unsupported_model_family_without_fallback() -> None:
    config = EngineConfig(
        model="tiny",
        model_kind="tiny-deepseek",
        disaggregation_mode="prefill-decode",
    )

    with pytest.raises(ValueError, match="not implemented for model kind 'tiny-deepseek'"):
        load_model_for_engine(config)


def test_disaggregation_rejects_paged_cache_before_cuda_topology(monkeypatch) -> None:
    config = EngineConfig(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        model_kind="llama3",
        cache_backend="paged",
        disaggregation_mode="prefill-decode",
    )
    monkeypatch.setattr(
        loader_module,
        "initialize_disaggregated_topology",
        lambda *_args, **_kwargs: pytest.fail("topology initialized before cache validation"),
    )

    with pytest.raises(ValueError, match="supports dense or flashinfer KV cache"):
        load_model_for_engine(config)


def test_deepseek_v4_disaggregation_rejects_misreported_flashinfer_cache(
    monkeypatch,
) -> None:
    config = EngineConfig(
        model="deepseek-ai/DeepSeek-V4-Flash",
        model_kind="deepseek-v4",
        cache_backend="flashinfer",
        disaggregation_mode="prefill-decode",
    )
    monkeypatch.setattr(
        loader_module,
        "initialize_disaggregated_topology",
        lambda *_args, **_kwargs: pytest.fail("topology initialized before cache validation"),
    )

    with pytest.raises(ValueError, match="requires --cache-backend dense"):
        load_model_for_engine(config)


def test_deepseek_v4_disaggregation_caps_model_cache_rows_before_load(monkeypatch) -> None:
    captured = {}
    config = EngineConfig(
        model="deepseek-ai/DeepSeek-V4-Flash",
        model_kind="deepseek-v4",
        max_batch_size=128,
        tensor_parallel_size=1,
        disaggregation_mode="prefill-decode",
    )
    topology = type(
        "Topology",
        (),
        {"role_group": object(), "device": torch.device("cpu")},
    )()

    class Evaluated:
        def eval(self):
            return self

    def from_pretrained(*args, **kwargs):
        del args
        captured["max_batch_size"] = kwargs["max_batch_size"]
        return Evaluated()

    monkeypatch.delenv("TORCHINFERNO_OPENAI_DISAGG_MAX_BATCH_SIZE", raising=False)
    monkeypatch.setattr(loader_module, "initialize_disaggregated_topology", lambda size: topology)
    monkeypatch.setattr(loader_module, "set_deepseek_v4_process_group", lambda group: None)
    monkeypatch.setattr(
        loader_module.DeepSeekV4TensorParallelForCausalLM,
        "from_pretrained",
        from_pretrained,
    )
    monkeypatch.setattr(
        loader_module,
        "DisaggregatedPrefillDecodeModel",
        lambda *args, **kwargs: Evaluated(),
    )

    load_model_for_engine(config)

    assert captured["max_batch_size"] == 8


@pytest.mark.parametrize(
    "selection",
    ({"device": "cuda:0"}, {"devices": ("cuda:2", "cuda:3")}),
)
def test_disaggregation_requires_cuda_visible_devices_for_gpu_selection(selection) -> None:
    config = EngineConfig(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        model_kind="llama3",
        disaggregation_mode="prefill-decode",
        **selection,
    )

    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        load_model_for_engine(config)


def test_deepseek_v4_tensor_parallel_requires_distributed_launch(monkeypatch) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    config = EngineConfig(
        model="deepseek-ai/DeepSeek-V4-Flash",
        model_kind="deepseek-v4",
        tensor_parallel_size=8,
    )

    with pytest.raises(RuntimeError, match="DeepSeek V4 tensor-parallel serving"):
        load_model_for_engine(config)


def test_standard_tensor_parallel_runtime_joins_all_ranks_and_broadcasts_sampling() -> None:
    context = torch.multiprocessing.get_context("spawn")
    results = context.Queue()
    world_size = 2
    port = _unused_tcp_port()
    processes = [
        context.Process(
            target=_standard_tp_runtime_worker,
            args=(rank, world_size, port, results),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    observed = sorted(results.get(timeout=2) for _ in range(world_size))
    assert observed == [(0, "cpu", 2, 7), (1, "cpu", 2, 7)]


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
