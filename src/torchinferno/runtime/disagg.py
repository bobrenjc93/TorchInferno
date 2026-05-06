from __future__ import annotations

from dataclasses import dataclass
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from typing import Any, Literal
from urllib import error, request

import torch

from torchinferno.models.dsv4 import DSv4Cache, DSv4ForCausalLM, sample_next_token, tiny_dsv4_config


RankRole = Literal["prefill", "decode"]


@dataclass(frozen=True)
class RankEndpoint:
    rank_id: int
    role: RankRole
    host: str
    port: int
    file: str

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class RankFilePlan:
    output_dir: Path
    endpoints: tuple[RankEndpoint, ...]
    manifest: Path
    client_smoke: Path


class JsonRankClient:
    """Small JSON-over-HTTP client for standalone rank files.

    This intentionally mirrors a gRPC-like request shape while staying
    dependency-free for local iteration. The transport can be swapped without
    changing rank method names or payload contracts.
    """

    def __init__(self, base_url: str, *, timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def call(self, method: str, **params: Any) -> dict[str, Any]:
        payload = json.dumps({"method": method, "params": params}).encode("utf-8")
        rpc_request = request.Request(
            f"{self.base_url}/rpc",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(rpc_request, timeout=self.timeout_s) as response:
                body = response.read()
        except error.HTTPError as exc:
            body = exc.read()
        message = json.loads(body.decode("utf-8"))
        if not message.get("ok", False):
            rpc_error = message.get("error", {})
            raise RuntimeError(f"{method} failed: {rpc_error.get('type')}: {rpc_error.get('message')}")
        return dict(message["result"])


class AgentRank:
    """Base class for one editable rank process."""

    role: RankRole

    def __init__(
        self,
        *,
        rank_id: int,
        role: RankRole,
        device: str = "cpu",
        seed: int = 0,
        vocab_size: int = 128,
        max_seq_len: int = 64,
    ) -> None:
        self.rank_id = rank_id
        self.role = role
        self.device = _normalize_device(torch.device(device))
        self.seed = seed
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

    def health(self) -> dict[str, Any]:
        return {
            "rank_id": self.rank_id,
            "role": self.role,
            "device": str(self.device),
            "seed": self.seed,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
        }

    def handle_rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "health":
            return self.health()
        handler = getattr(self, method, None)
        if handler is None or method.startswith("_"):
            raise ValueError(f"unknown rank RPC method: {method}")
        result = handler(**params)
        if not isinstance(result, dict):
            raise TypeError(f"rank RPC method {method} must return a dict")
        return result


class DSv4Rank(AgentRank):
    def __init__(
        self,
        *,
        rank_id: int,
        role: RankRole,
        device: str = "cpu",
        seed: int = 0,
        vocab_size: int = 128,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__(
            rank_id=rank_id,
            role=role,
            device=device,
            seed=seed,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
        )
        self.model = self._build_model()

    def _build_model(self) -> DSv4ForCausalLM:
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        model = DSv4ForCausalLM(tiny_dsv4_config(vocab_size=self.vocab_size, max_seq_len=self.max_seq_len))
        return model.to(self.device).eval()


class DSv4PrefillRank(DSv4Rank):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(role="prefill", **kwargs)

    @torch.inference_mode()
    def prefill(
        self,
        *,
        request_id: str,
        prompt: list[int],
        max_new_tokens: int,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        if not prompt:
            raise ValueError("prompt must contain at least one token")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        input_ids = torch.tensor([prompt], device=self.device, dtype=torch.long)
        cache = self.model.allocate_cache(
            1,
            min(self.max_seq_len, len(prompt) + max(max_new_tokens, 1)),
            device=self.device,
        )
        logits, cache = self.model(input_ids, cache=cache, use_cache=True)
        if max_new_tokens == 0:
            return {
                "request_id": request_id,
                "finished": True,
                "tokens": prompt,
                "transfer": None,
            }
        next_token = int(sample_next_token(logits[:, -1, :], temperature).item())
        transfer = _serialize_dsv4_transfer(
            request_id=request_id,
            prefill_rank_id=self.rank_id,
            prompt=prompt,
            next_token=next_token,
            max_seq_len=cache.layers[0].max_seq_len,
            cache=cache,
        )
        return {
            "request_id": request_id,
            "finished": False,
            "tokens": [*prompt, next_token],
            "transfer": transfer,
        }


class DSv4DecodeRank(DSv4Rank):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(role="decode", **kwargs)

    @torch.inference_mode()
    def decode(
        self,
        *,
        transfer: dict[str, Any] | None = None,
        tokens: list[int] | None = None,
        max_new_tokens: int,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if transfer is None:
            if tokens is None:
                raise ValueError("decode requires transfer or finished tokens")
            return {"request_id": "", "tokens": tokens, "generated_tokens": 0}
        cache = _deserialize_dsv4_cache(self.model, transfer, self.device)
        output_tokens = [*transfer["prompt"], int(transfer["next_token"])]
        next_token = torch.tensor([[output_tokens[-1]]], device=self.device, dtype=torch.long)
        for _ in range(1, max_new_tokens):
            logits, cache = self.model(next_token, cache=cache, use_cache=True)
            sampled = sample_next_token(logits[:, -1, :], temperature)
            token = int(sampled.item())
            output_tokens.append(token)
            next_token = sampled[:, None]
        return {
            "request_id": transfer["request_id"],
            "tokens": output_tokens,
            "generated_tokens": max_new_tokens,
            "source_prefill_rank": transfer["prefill_rank_id"],
            "decode_rank_id": self.rank_id,
        }


class _RankHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], rank: AgentRank) -> None:
        super().__init__(server_address, _RankRequestHandler)
        self.rank = rank


class _RankRequestHandler(BaseHTTPRequestHandler):
    server: _RankHTTPServer

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send_json({"ok": False, "error": {"type": "NotFound", "message": self.path}}, status=404)
            return
        self._send_json({"ok": True, "result": self.server.rank.health()})

    def do_POST(self) -> None:
        if self.path != "/rpc":
            self._send_json({"ok": False, "error": {"type": "NotFound", "message": self.path}}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = self.server.rank.handle_rpc(str(payload["method"]), dict(payload.get("params", {})))
            self._send_json({"ok": True, "result": result})
        except Exception as exc:
            self._send_json(
                {"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}},
                status=500,
            )

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve_rank(rank: AgentRank, *, host: str = "127.0.0.1", port: int) -> None:
    server = _RankHTTPServer((host, port), rank)
    print(f"rank_id={rank.rank_id} role={rank.role} url=http://{host}:{server.server_port}", flush=True)
    server.serve_forever()


def start_rank_server(rank: AgentRank, *, host: str = "127.0.0.1", port: int = 0) -> tuple[_RankHTTPServer, threading.Thread]:
    server = _RankHTTPServer((host, port), rank)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def run_disagg_request(
    *,
    prefill_url: str,
    decode_url: str,
    request_id: str,
    prompt: list[int],
    max_new_tokens: int,
    temperature: float = 0.0,
    timeout_s: float = 30.0,
) -> dict[str, Any]:
    prefill = JsonRankClient(prefill_url, timeout_s=timeout_s)
    decode = JsonRankClient(decode_url, timeout_s=timeout_s)
    prefill_result = prefill.call(
        "prefill",
        request_id=request_id,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    decode_result = decode.call(
        "decode",
        transfer=prefill_result.get("transfer"),
        tokens=prefill_result.get("tokens"),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return {"prefill": prefill_result, "decode": decode_result}


def write_rank_files(
    output_dir: Path,
    *,
    prefill_ranks: int,
    decode_ranks: int,
    host: str = "127.0.0.1",
    base_port: int = 8800,
    device: str = "cpu",
    seed: int = 0,
    vocab_size: int = 128,
    max_seq_len: int = 64,
) -> RankFilePlan:
    if prefill_ranks < 1 or decode_ranks < 1:
        raise ValueError("prefill_ranks and decode_ranks must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    endpoints: list[RankEndpoint] = []
    rank_id = 0
    for _ in range(prefill_ranks):
        port = base_port + rank_id
        filename = f"rank_{rank_id}_prefill.py"
        _write_rank_file(
            output_dir / filename,
            rank_id=rank_id,
            role="prefill",
            host=host,
            port=port,
            device=device,
            seed=seed,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
        )
        endpoints.append(RankEndpoint(rank_id, "prefill", host, port, filename))
        rank_id += 1
    for _ in range(decode_ranks):
        port = base_port + rank_id
        filename = f"rank_{rank_id}_decode.py"
        _write_rank_file(
            output_dir / filename,
            rank_id=rank_id,
            role="decode",
            host=host,
            port=port,
            device=device,
            seed=seed,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
        )
        endpoints.append(RankEndpoint(rank_id, "decode", host, port, filename))
        rank_id += 1
    manifest = output_dir / "ranks.json"
    client_smoke = output_dir / "client_smoke.py"
    _write_json(
        manifest,
        {
            "endpoints": [
                {
                    "rank_id": endpoint.rank_id,
                    "role": endpoint.role,
                    "host": endpoint.host,
                    "port": endpoint.port,
                    "url": endpoint.url,
                    "file": endpoint.file,
                }
                for endpoint in endpoints
            ]
        },
    )
    _write_client_smoke(client_smoke, endpoints)
    _write_readme(output_dir / "README.md", endpoints)
    return RankFilePlan(output_dir, tuple(endpoints), manifest, client_smoke)


def _write_rank_file(
    path: Path,
    *,
    rank_id: int,
    role: RankRole,
    host: str,
    port: int,
    device: str,
    seed: int,
    vocab_size: int,
    max_seq_len: int,
) -> None:
    cls = "DSv4PrefillRank" if role == "prefill" else "DSv4DecodeRank"
    repo_root = str(Path.cwd())
    source = f'''#!/usr/bin/env python3
"""Agent-editable TorchInferno {role} rank.

This file is intentionally standalone. Optimize this rank here, then keep the
RPC method contract stable so wrappers and schedulers can keep calling it.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path({repo_root!r})
if (REPO_ROOT / "src").exists():
    sys.path.insert(0, str(REPO_ROOT / "src"))

from torchinferno.runtime.disagg import {cls}, serve_rank


RANK_ID = {rank_id}
HOST = {host!r}
PORT = {port}
DEVICE = {device!r}
SEED = {seed}
VOCAB_SIZE = {vocab_size}
MAX_SEQ_LEN = {max_seq_len}


def build_rank():
    return {cls}(
        rank_id=RANK_ID,
        device=DEVICE,
        seed=SEED,
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
    )


if __name__ == "__main__":
    serve_rank(build_rank(), host=HOST, port=PORT)
'''
    path.write_text(source)
    path.chmod(0o755)


def _write_client_smoke(path: Path, endpoints: list[RankEndpoint]) -> None:
    prefill = next(endpoint for endpoint in endpoints if endpoint.role == "prefill")
    decode = next(endpoint for endpoint in endpoints if endpoint.role == "decode")
    repo_root = str(Path.cwd())
    source = f'''#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path({repo_root!r})
if (REPO_ROOT / "src").exists():
    sys.path.insert(0, str(REPO_ROOT / "src"))

from torchinferno.runtime.disagg import run_disagg_request


if __name__ == "__main__":
    result = run_disagg_request(
        prefill_url={prefill.url!r},
        decode_url={decode.url!r},
        request_id="req-smoke",
        prompt=[1, 2, 3],
        max_new_tokens=2,
    )
    print(result)
'''
    path.write_text(source)
    path.chmod(0o755)


def _write_readme(path: Path, endpoints: list[RankEndpoint]) -> None:
    commands = "\n".join(f"python3 {endpoint.file}" for endpoint in endpoints)
    smoke = "python3 client_smoke.py"
    path.write_text(
        "# TorchInferno Disaggregated Rank Files\n\n"
        "Start each rank in its own terminal:\n\n"
        "```bash\n"
        f"{commands}\n"
        "```\n\n"
        "Then send one request through the generated client:\n\n"
        "```bash\n"
        f"{smoke}\n"
        "```\n"
    )


def _serialize_dsv4_transfer(
    *,
    request_id: str,
    prefill_rank_id: int,
    prompt: list[int],
    next_token: int,
    max_seq_len: int,
    cache: DSv4Cache,
) -> dict[str, Any]:
    layers = []
    for layer in cache.layers:
        seq_len = layer.seq_len
        layers.append(
            {
                "seq_len": seq_len,
                "keys": layer.keys[:1, :, :seq_len, :].detach().cpu().tolist(),
                "values": layer.values[:1, :, :seq_len, :].detach().cpu().tolist(),
            }
        )
    return {
        "kind": "torchinferno.dsv4.kv.v1",
        "request_id": request_id,
        "prefill_rank_id": prefill_rank_id,
        "prompt": prompt,
        "next_token": next_token,
        "max_seq_len": max_seq_len,
        "layers": layers,
    }


def _deserialize_dsv4_cache(model: DSv4ForCausalLM, transfer: dict[str, Any], device: torch.device) -> DSv4Cache:
    if transfer.get("kind") != "torchinferno.dsv4.kv.v1":
        raise ValueError(f"unsupported transfer kind: {transfer.get('kind')}")
    cache = model.allocate_cache(
        1,
        int(transfer["max_seq_len"]),
        device=device,
        dtype=model.embed_tokens.weight.dtype,
    )
    for layer, state in zip(cache.layers, transfer["layers"]):
        seq_len = int(state["seq_len"])
        keys = torch.tensor(state["keys"], device=device, dtype=layer.keys.dtype)
        values = torch.tensor(state["values"], device=device, dtype=layer.values.dtype)
        layer.keys[:1, :, :seq_len, :].copy_(keys)
        layer.values[:1, :, :seq_len, :].copy_(values)
        layer.seq_len = seq_len
    return cache


def _normalize_device(device: torch.device) -> torch.device:
    if device.type != "cuda":
        return device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    index = torch.cuda.current_device() if device.index is None else device.index
    torch.cuda.set_device(index)
    return torch.device("cuda", index)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
