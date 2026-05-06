import json
import os
import subprocess
import sys

import torch

from torchinferno.models.dsv4 import DSv4ForCausalLM, tiny_dsv4_config
from torchinferno.runtime.disagg import (
    DSv4DecodeRank,
    DSv4PrefillRank,
    JsonRankClient,
    run_disagg_request,
    start_rank_server,
    write_rank_files,
)


def test_dsv4_disagg_rank_roundtrip_matches_full_generate() -> None:
    prompt = [1, 2, 3]
    max_new_tokens = 3
    seed = 60
    prefill = DSv4PrefillRank(rank_id=0, device="cpu", seed=seed, vocab_size=32, max_seq_len=16)
    decode = DSv4DecodeRank(rank_id=1, device="cpu", seed=seed, vocab_size=32, max_seq_len=16)

    prefill_result = prefill.prefill(
        request_id="req",
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )
    decode_result = decode.decode(
        transfer=prefill_result["transfer"],
        max_new_tokens=max_new_tokens,
    )

    torch.manual_seed(seed)
    model = DSv4ForCausalLM(tiny_dsv4_config(vocab_size=32, max_seq_len=16)).eval()
    with torch.inference_mode():
        expected = model.generate(torch.tensor([prompt], dtype=torch.long), max_new_tokens=max_new_tokens)

    assert decode_result["tokens"] == expected[0].tolist()
    assert decode_result["source_prefill_rank"] == 0
    assert decode_result["decode_rank_id"] == 1


def test_json_rank_server_roundtrip() -> None:
    seed = 61
    prefill = DSv4PrefillRank(rank_id=0, device="cpu", seed=seed, vocab_size=32, max_seq_len=16)
    decode = DSv4DecodeRank(rank_id=1, device="cpu", seed=seed, vocab_size=32, max_seq_len=16)
    prefill_server, _ = start_rank_server(prefill)
    decode_server, _ = start_rank_server(decode)
    try:
        prefill_url = f"http://127.0.0.1:{prefill_server.server_port}"
        decode_url = f"http://127.0.0.1:{decode_server.server_port}"
        health = JsonRankClient(prefill_url).call("health")
        result = run_disagg_request(
            prefill_url=prefill_url,
            decode_url=decode_url,
            request_id="req",
            prompt=[1, 2, 3],
            max_new_tokens=2,
        )
    finally:
        prefill_server.shutdown()
        decode_server.shutdown()
        prefill_server.server_close()
        decode_server.server_close()

    assert health["role"] == "prefill"
    assert result["decode"]["request_id"] == "req"
    assert len(result["decode"]["tokens"]) == 5


def test_disagg_init_writes_agent_editable_rank_files(tmp_path) -> None:
    plan = write_rank_files(
        tmp_path,
        prefill_ranks=2,
        decode_ranks=2,
        base_port=8900,
        device="cpu",
        seed=62,
        vocab_size=32,
        max_seq_len=16,
    )

    manifest = json.loads(plan.manifest.read_text())

    assert len(plan.endpoints) == 4
    assert len(manifest["endpoints"]) == 4
    assert (tmp_path / "rank_0_prefill.py").exists()
    assert (tmp_path / "rank_2_decode.py").exists()
    assert "Agent-editable TorchInferno prefill rank" in (tmp_path / "rank_0_prefill.py").read_text()
    assert plan.client_smoke.exists()


def test_disagg_init_cli_writes_rank_files(tmp_path) -> None:
    env = {**os.environ, "PYTHONPATH": "src"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchinferno.cli",
            "disagg-init",
            str(tmp_path),
            "--prefill-ranks",
            "1",
            "--decode-ranks",
            "1",
            "--base-port",
            "8910",
            "--device",
            "cpu",
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    assert "TorchInferno disaggregated rank files" in result.stdout
    assert (tmp_path / "rank_0_prefill.py").exists()
    assert (tmp_path / "rank_1_decode.py").exists()
