from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class ModelConfig:
    model: str
    model_kind: str = "auto"
    tokenizer: str | None = None
    tensor_parallel_size: int = 1
    devices: tuple[str, ...] = ()
    device: str | None = None
    dtype: str = "auto"
    max_model_len: int | None = None
    trust_remote_code: bool = False
    token: str | None = None
    revision: str | None = None
    cache_dir: str | Path | None = None
    llama_parallelism: str = "auto"
    disaggregation_mode: str = "none"
    disaggregation_profile: bool = False


@dataclass(frozen=True)
class CacheConfig:
    backend: str = "dense"
    page_size: int = 16


@dataclass(frozen=True)
class SchedulerConfig:
    max_batch_size: int = 64
    batch_wait_ms: float = 10.0
    single_request_admission_wait_ms: float | None = None


@dataclass(frozen=True)
class EngineConfig:
    model: str
    model_kind: str = "auto"
    tokenizer: str | None = None
    tensor_parallel_size: int = 1
    devices: tuple[str, ...] = ()
    device: str | None = None
    dtype: str = "auto"
    max_model_len: int | None = None
    trust_remote_code: bool = False
    token: str | None = None
    revision: str | None = None
    cache_dir: str | Path | None = None
    cache_backend: str = "dense"
    page_size: int = 16
    max_batch_size: int = 64
    batch_wait_ms: float = 10.0
    single_request_admission_wait_ms: float | None = None
    llama_parallelism: str = "auto"
    disaggregation_mode: str = "none"
    disaggregation_profile: bool = False

    @classmethod
    def from_parts(
        cls,
        model: ModelConfig,
        *,
        cache: CacheConfig | None = None,
        scheduler: SchedulerConfig | None = None,
    ) -> "EngineConfig":
        cache = CacheConfig() if cache is None else cache
        scheduler = SchedulerConfig() if scheduler is None else scheduler
        return cls(
            model=model.model,
            model_kind=model.model_kind,
            tokenizer=model.tokenizer,
            tensor_parallel_size=model.tensor_parallel_size,
            devices=model.devices,
            device=model.device,
            dtype=model.dtype,
            max_model_len=model.max_model_len,
            trust_remote_code=model.trust_remote_code,
            token=model.token,
            revision=model.revision,
            cache_dir=model.cache_dir,
            cache_backend=cache.backend,
            page_size=cache.page_size,
            max_batch_size=scheduler.max_batch_size,
            batch_wait_ms=scheduler.batch_wait_ms,
            single_request_admission_wait_ms=scheduler.single_request_admission_wait_ms,
            llama_parallelism=model.llama_parallelism,
            disaggregation_mode=model.disaggregation_mode,
            disaggregation_profile=model.disaggregation_profile,
        )

    @classmethod
    def from_legacy_openai(cls, config: object) -> "EngineConfig":
        return cls(
            model=str(getattr(config, "model")),
            model_kind=str(getattr(config, "model_kind", "auto")),
            tokenizer=getattr(config, "tokenizer", None),
            tensor_parallel_size=int(getattr(config, "tensor_parallel_size", 1)),
            devices=tuple(getattr(config, "devices", ()) or ()),
            device=getattr(config, "device", None),
            dtype=str(getattr(config, "dtype", "auto")),
            max_model_len=getattr(config, "max_model_len", None),
            trust_remote_code=bool(getattr(config, "trust_remote_code", False)),
            token=getattr(config, "token", None),
            revision=getattr(config, "revision", None),
            cache_dir=getattr(config, "cache_dir", None),
            cache_backend=str(getattr(config, "cache_backend", "dense")),
            page_size=int(getattr(config, "page_size", 16)),
            max_batch_size=int(getattr(config, "max_batch_size", 64)),
            batch_wait_ms=float(getattr(config, "batch_wait_ms", 10.0)),
            single_request_admission_wait_ms=getattr(config, "single_request_admission_wait_ms", None),
            llama_parallelism=str(getattr(config, "llama_parallelism", "auto")),
            disaggregation_mode=str(getattr(config, "disaggregation_mode", "none")),
            disaggregation_profile=bool(getattr(config, "disaggregation_profile", False)),
        )

    @classmethod
    def from_legacy_args(cls, args: object) -> "EngineConfig":
        devices: Sequence[str] = ()
        raw_devices = getattr(args, "devices", None)
        if raw_devices:
            devices = tuple(part.strip() for part in str(raw_devices).split(",") if part.strip())
        return cls(
            model=str(getattr(args, "model")),
            model_kind=str(getattr(args, "model_kind", "auto")),
            tokenizer=getattr(args, "tokenizer", None),
            tensor_parallel_size=int(getattr(args, "tensor_parallel_size", 1)),
            devices=tuple(devices),
            device=getattr(args, "device", None),
            dtype=str(getattr(args, "dtype", "auto")),
            max_model_len=getattr(args, "max_model_len", None),
            trust_remote_code=bool(getattr(args, "trust_remote_code", False)),
            token=getattr(args, "token", None),
            revision=getattr(args, "revision", None),
            cache_dir=getattr(args, "cache_dir", None),
            cache_backend=str(getattr(args, "cache_backend", "dense")),
            page_size=int(getattr(args, "page_size", 16)),
            max_batch_size=int(getattr(args, "max_batch_size", 64)),
            batch_wait_ms=float(getattr(args, "batch_wait_ms", 10.0)),
            single_request_admission_wait_ms=getattr(args, "single_request_admission_wait_ms", None),
            llama_parallelism=str(getattr(args, "llama_parallelism", "auto")),
            disaggregation_mode=str(getattr(args, "disaggregation_mode", "none")),
            disaggregation_profile=bool(getattr(args, "disaggregation_profile", False)),
        )

    def to_legacy_openai_config(self, *, host: str = "0.0.0.0", port: int = 8000) -> object:
        from torchinferno.openai_server import OpenAIServerConfig

        return OpenAIServerConfig(
            model=self.model,
            host=host,
            port=port,
            model_kind=self.model_kind,
            tokenizer=self.tokenizer,
            tensor_parallel_size=self.tensor_parallel_size,
            devices=self.devices,
            device=self.device,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            trust_remote_code=self.trust_remote_code,
            token=self.token,
            revision=self.revision,
            cache_dir=str(self.cache_dir) if self.cache_dir is not None else None,
            cache_backend=self.cache_backend,
            page_size=self.page_size,
            max_batch_size=self.max_batch_size,
            batch_wait_ms=self.batch_wait_ms,
            single_request_admission_wait_ms=self.single_request_admission_wait_ms,
            llama_parallelism=self.llama_parallelism,
            disaggregation_mode=self.disaggregation_mode,
            disaggregation_profile=self.disaggregation_profile,
        )
