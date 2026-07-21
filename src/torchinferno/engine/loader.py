from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Sequence

import torch

from torchinferno.models.auto import load_model_auto
from torchinferno.models.deepseek_v32 import DeepSeekV32ForCausalLM, tiny_deepseek_v32_config
from torchinferno.models.dsv4 import DSv4ForCausalLM, tiny_dsv4_config
from torchinferno.models.llama3 import Llama3V0ForCausalLM, tiny_llama3_v0_config
from torchinferno.models.llama3.pipeline import Llama3PipelineForCausalLM
from torchinferno.models.llama3.tensor_parallel import (
    Llama3TensorParallelForCausalLM,
    set_tensor_parallel_process_group,
    validate_symm_mem_allreduce_collective,
)
from torchinferno.runtime.disaggregated import (
    DisaggregatedPrefillDecodeModel,
    initialize_disaggregated_topology,
)


def load_model_for_engine(config: object) -> tuple[object, torch.device]:
    kind = infer_model_kind(config)
    configured_disaggregation = disaggregation_mode(config)
    if configured_disaggregation != "none" and kind != "llama3":
        raise ValueError(
            f"disaggregated prefill/decode is not implemented for model kind {kind!r}"
        )
    if configured_disaggregation == "prefill-decode":
        cache_backend = str(getattr(config, "cache_backend", "dense")).strip().lower()
        if cache_backend not in {"dense", "flashinfer"}:
            raise ValueError(
                "disaggregated prefill/decode supports dense or flashinfer KV cache"
            )
    dtype = resolve_dtype(str(getattr(config, "dtype", "auto")))
    if kind == "tiny-deepseek":
        device = primary_device(config)
        model = DeepSeekV32ForCausalLM(tiny_deepseek_v32_config(max_position_embeddings=_max_model_len(config) or 128))
        return model.to(device=device, dtype=dtype or torch.float32).eval(), device
    if kind == "tiny-dsv4":
        device = primary_device(config)
        model = DSv4ForCausalLM(tiny_dsv4_config(max_seq_len=_max_model_len(config) or 128))
        return model.to(device=device, dtype=dtype or torch.float32).eval(), device
    if kind == "tiny-llama3":
        device = primary_device(config)
        model = Llama3V0ForCausalLM(tiny_llama3_v0_config(max_position_embeddings=_max_model_len(config) or 128))
        return model.to(device=device, dtype=dtype or torch.float32).eval(), device
    if kind == "llama3":
        if configured_disaggregation == "prefill-decode":
            if str(getattr(config, "llama_parallelism", "auto")).lower() == "pipeline":
                raise ValueError("disaggregated prefill/decode requires Llama tensor parallelism")
            topology = initialize_disaggregated_topology(
                int(getattr(config, "tensor_parallel_size", 1))
            )
            set_tensor_parallel_process_group(topology.role_group)
            try:
                role_model = Llama3TensorParallelForCausalLM.from_pretrained(
                    str(getattr(config, "model")),
                    dtype=str(getattr(config, "dtype", "auto")),
                    token=getattr(config, "token", None),
                    revision=getattr(config, "revision", None),
                    cache_dir=getattr(config, "cache_dir", None),
                ).eval()
                validate_symm_mem_allreduce_collective(role_model, topology.device)
                model = DisaggregatedPrefillDecodeModel(
                    role_model,
                    topology,
                    cache_backend=str(getattr(config, "cache_backend", "dense")),
                    page_size=int(getattr(config, "page_size", 16)),
                    profile_transfer=bool(getattr(config, "disaggregation_profile", False)),
                ).eval()
            except BaseException:
                set_tensor_parallel_process_group(None)
                raise
            return model, topology.device
        if llama_parallelism(config) == "tensor":
            if int(getattr(config, "tensor_parallel_size", 1)) > 1 and not distributed_env_requested():
                raise RuntimeError(
                    "Llama tensor parallel serving requires a distributed launch. "
                    "Use torchrun, or start torchinferno.openai_server normally with "
                    "--tensor-parallel-size > 1 so it can auto-launch workers."
                )
            model = Llama3TensorParallelForCausalLM.from_pretrained(
                str(getattr(config, "model")),
                dtype=str(getattr(config, "dtype", "auto")),
                token=getattr(config, "token", None),
                revision=getattr(config, "revision", None),
                cache_dir=getattr(config, "cache_dir", None),
            ).eval()
            return model, model.device
        devices = server_devices(config)
        model = Llama3PipelineForCausalLM.from_pretrained(
            str(getattr(config, "model")),
            devices=devices,
            dtype=str(getattr(config, "dtype", "auto")),
            token=getattr(config, "token", None),
            revision=getattr(config, "revision", None),
            cache_dir=getattr(config, "cache_dir", None),
        ).eval()
        return model, torch.device(devices[0])
    device = primary_device(config)
    model = load_model_auto(
        str(getattr(config, "model")),
        token=getattr(config, "token", None),
        revision=getattr(config, "revision", None),
        cache_dir=getattr(config, "cache_dir", None),
        map_location=device,
        strict=True,
    )
    if dtype is not None:
        model = model.to(dtype=dtype)
    return model.to(device).eval(), device


def infer_model_kind(config: object) -> str:
    kind = str(getattr(config, "model_kind", "auto")).lower()
    if kind != "auto":
        return kind
    model = str(getattr(config, "model")).lower()
    if "llama" in model:
        return "llama3"
    path = Path(str(getattr(config, "model"))).expanduser()
    config_path = path / "config.json"
    if config_path.exists():
        data = json.loads(config_path.read_text())
        model_type = str(data.get("model_type", "")).lower()
        if "llama" in model_type:
            return "llama3"
        if "deepseek" in model_type:
            return "deepseek"
        if model_type == "dsv4":
            return "dsv4"
    return "auto"


def primary_device(config: object) -> torch.device:
    device = getattr(config, "device", None)
    if device:
        return torch.device(str(device))
    devices = server_devices(config)
    return torch.device(devices[0])


def server_devices(config: object) -> tuple[str, ...]:
    devices = tuple(getattr(config, "devices", ()) or ())
    if devices:
        return tuple(str(device) for device in devices)
    device = getattr(config, "device", None)
    if device:
        return (str(device),)
    if torch.cuda.is_available():
        count = max(1, min(int(getattr(config, "tensor_parallel_size", 1)), torch.cuda.device_count()))
        return tuple(f"cuda:{idx}" for idx in range(count))
    return ("cpu",)


def llama_parallelism(config: object) -> str:
    mode = str(getattr(config, "llama_parallelism", "auto")).lower()
    if mode == "pipeline":
        return "pipeline"
    if mode == "tensor":
        return "tensor"
    if mode != "auto":
        raise ValueError(f"unsupported llama parallelism: {getattr(config, 'llama_parallelism', mode)}")
    if distributed_env_requested() or int(getattr(config, "tensor_parallel_size", 1)) > 1:
        return "tensor"
    return "pipeline"


def disaggregation_mode(config: object) -> str:
    mode = str(getattr(config, "disaggregation_mode", "none")).strip().lower()
    if mode not in {"none", "prefill-decode"}:
        raise ValueError(f"unsupported disaggregation mode: {getattr(config, 'disaggregation_mode', mode)}")
    if mode == "prefill-decode" and (
        getattr(config, "device", None) or tuple(getattr(config, "devices", ()) or ())
    ):
        raise ValueError(
            "disaggregated prefill/decode maps LOCAL_RANK to visible GPUs; "
            "select devices with CUDA_VISIBLE_DEVICES instead of --device/--devices"
        )
    return mode


def distributed_env_requested() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def should_reexec_distributed_server(config: object) -> bool:
    if os.environ.get("TORCHINFERNO_OPENAI_AUTO_TORCHRUN", "1") == "0":
        return False
    if distributed_env_requested():
        return False
    if infer_model_kind(config) != "llama3":
        return False
    if disaggregation_mode(config) == "prefill-decode":
        return True
    if int(getattr(config, "tensor_parallel_size", 1)) <= 1:
        return False
    return str(getattr(config, "llama_parallelism", "auto")).lower() != "pipeline"


def distributed_server_command(config: object, argv: Sequence[str]) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
    ]
    configured_rdzv_endpoint = os.environ.get("TORCHINFERNO_TORCHRUN_RDZV_ENDPOINT")
    if configured_rdzv_endpoint:
        command.extend(
            [
                "--rdzv-backend",
                "c10d",
                "--rdzv-endpoint",
                configured_rdzv_endpoint,
                "--rdzv-id",
                _torchrun_rdzv_id(configured_rdzv_endpoint),
                "--rdzv-conf",
                "is_host=true",
            ]
        )
    else:
        command.append("--standalone")
    command.extend(
        [
            "--nproc-per-node",
            str(distributed_server_world_size(config)),
            "-m",
            "torchinferno.openai_server",
            *argv,
        ]
    )
    return command


def distributed_server_world_size(config: object) -> int:
    tensor_parallel_size = int(getattr(config, "tensor_parallel_size", 1))
    if tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be positive")
    if disaggregation_mode(config) == "prefill-decode":
        return 2 * tensor_parallel_size
    return tensor_parallel_size


def _torchrun_rdzv_id(rdzv_endpoint: str) -> str:
    port = rdzv_endpoint.rsplit(":", 1)[-1]
    return f"torchinferno-openai-{os.getpid()}-{port}"


def resolve_dtype(dtype: str) -> torch.dtype | None:
    normalized = dtype.lower().replace("torch.", "")
    if normalized == "auto":
        return None
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {dtype}")


def _max_model_len(config: object) -> int | None:
    value = getattr(config, "max_model_len", None)
    return int(value) if value is not None else None
