from __future__ import annotations

from pathlib import Path

import torchinferno.models.llama3.pipeline as llama3_pipeline


def test_resolve_llama3_checkpoint_uses_complete_hf_cache(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "hub"
    snapshot = _write_hf_snapshot_cache(cache_dir, "org/model")

    def snapshot_download(*args: object, **kwargs: object) -> str:
        raise AssertionError("complete cached snapshots should not hit the network")

    monkeypatch.setattr(llama3_pipeline, "snapshot_download", snapshot_download)

    resolved = llama3_pipeline.resolve_llama3_checkpoint("org/model", cache_dir=cache_dir)

    assert resolved == snapshot


def test_resolve_llama3_checkpoint_downloads_when_hf_cache_is_incomplete(
    tmp_path,
    monkeypatch,
) -> None:
    cache_dir = tmp_path / "hub"
    _write_hf_snapshot_cache(cache_dir, "org/model", include_index=False)
    downloaded = tmp_path / "downloaded"
    calls: list[dict[str, object]] = []

    def snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(downloaded)

    monkeypatch.setattr(llama3_pipeline, "snapshot_download", snapshot_download)

    resolved = llama3_pipeline.resolve_llama3_checkpoint(
        "org/model",
        token="hf-token",
        revision="main",
        cache_dir=cache_dir,
    )

    assert resolved == downloaded
    assert calls == [
        {
            "repo_id": "org/model",
            "revision": "main",
            "token": "hf-token",
            "cache_dir": str(cache_dir),
            "allow_patterns": [
                "config.json",
                "model.safetensors.index.json",
                "*.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "generation_config.json",
            ],
        }
    ]


def _write_hf_snapshot_cache(
    cache_dir: Path,
    repo_id: str,
    *,
    include_index: bool = True,
) -> Path:
    commit = "a" * 40
    repo_cache = cache_dir / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_cache / "snapshots" / commit
    snapshot.mkdir(parents=True)
    refs = repo_cache / "refs"
    refs.mkdir()
    (refs / "main").write_text(commit)
    (snapshot / "config.json").write_text("{}\n")
    if include_index:
        (snapshot / "model.safetensors.index.json").write_text('{"weight_map": {}}\n')
    return snapshot
