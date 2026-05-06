import json
import os
import subprocess
import sys

from torchinferno.profiling import ProfileRunConfig, run_profile_capture


def test_profile_capture_writes_artifacts_and_repro(tmp_path) -> None:
    output_dir = tmp_path / "profile"

    artifacts = run_profile_capture(
        ProfileRunConfig(
            output_dir=output_dir,
            device="cpu",
            batch_size=1,
            prompt_tokens=2,
            new_tokens=1,
            warmup=0,
        )
    )

    assert artifacts.manifest.exists()
    assert artifacts.repro.exists()
    assert artifacts.graph_json is not None and artifacts.graph_json.exists()
    assert artifacts.graph_text is not None and artifacts.graph_text.exists()
    assert artifacts.graph_code is not None and artifacts.graph_code.exists()
    assert artifacts.operator_profile is not None and artifacts.operator_profile.exists()
    assert artifacts.chrome_trace is not None and artifacts.chrome_trace.exists()

    output = json.loads(artifacts.output.read_text())
    graph = json.loads(artifacts.graph_json.read_text())
    profile = json.loads(artifacts.operator_profile.read_text())
    memory = json.loads(artifacts.memory_profile.read_text())

    assert output["shape"] == [1, 3]
    assert graph["node_count"] > 0
    assert profile["events"]
    assert memory["after"]["device"] == "cpu"

    repro_output = tmp_path / "repro_output.json"
    subprocess.run(
        [sys.executable, str(artifacts.repro), "--device", "cpu", "--output", str(repro_output)],
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(repro_output.read_text())["tokens"] == output["tokens"]


def test_profile_run_cli_writes_manifest(tmp_path) -> None:
    output_dir = tmp_path / "cli-profile"
    env = {**os.environ, "PYTHONPATH": "src"}

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchinferno.cli",
            "profile-run",
            str(output_dir),
            "--device",
            "cpu",
            "--batch-size",
            "1",
            "--prompt-tokens",
            "2",
            "--new-tokens",
            "1",
            "--warmup",
            "0",
            "--no-profiler",
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert "TorchInferno profile run" in result.stdout
    assert manifest["artifacts"]["operator_profile"] is None
    assert manifest["artifacts"]["graph_json"] == "graph.json"
