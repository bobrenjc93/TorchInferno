import json
import os
import subprocess
import sys

from torchinferno.profiling import (
    PatternProfileConfig,
    ProfileRunConfig,
    RegionProfileConfig,
    SubgraphProfileConfig,
    run_pattern_profile_capture,
    run_profile_capture,
    run_region_profile_capture,
    run_subgraph_profile_capture,
)


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


def test_region_profile_capture_writes_focused_artifacts_and_repro(tmp_path) -> None:
    output_dir = tmp_path / "region-profile"

    artifacts = run_region_profile_capture(
        RegionProfileConfig(
            output_dir=output_dir,
            region="layers.0.attn",
            device="cpu",
            batch_size=1,
            tokens=3,
            vocab_size=32,
            warmup=0,
            iters=1,
        )
    )

    assert artifacts.manifest.exists()
    assert artifacts.repro.exists()
    assert artifacts.graph_json is not None and artifacts.graph_json.exists()
    assert artifacts.operator_profile is not None and artifacts.operator_profile.exists()
    assert artifacts.chrome_trace is not None and artifacts.chrome_trace.exists()

    region_spec = json.loads((output_dir / "region_spec.json").read_text())
    graph = json.loads(artifacts.graph_json.read_text())
    profile = json.loads(artifacts.operator_profile.read_text())
    output = json.loads(artifacts.output.read_text())

    assert region_spec["resolved_module_path"] == "layers.0.attn"
    assert graph["node_count"] > 0
    assert profile["events"]
    assert output["output"]["shape"] == [1, 3, 64]

    subprocess.run(
        [
            sys.executable,
            str(artifacts.repro),
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path / "region-repro"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert (tmp_path / "region-repro" / "manifest.json").exists()


def test_pattern_profile_capture_writes_pass_comparison_and_repro(tmp_path) -> None:
    output_dir = tmp_path / "pattern-profile"

    artifacts = run_pattern_profile_capture(
        PatternProfileConfig(
            output_dir=output_dir,
            device="cpu",
            batch_size=1,
            tokens=3,
            hidden_size=16,
            warmup=0,
            iters=1,
        )
    )

    assert artifacts.manifest.exists()
    assert artifacts.repro.exists()
    assert artifacts.reference_graph is not None and artifacts.reference_graph.exists()
    assert artifacts.optimized_graph is not None and artifacts.optimized_graph.exists()
    assert artifacts.reference_profile is not None and artifacts.reference_profile.exists()
    assert artifacts.optimized_profile is not None and artifacts.optimized_profile.exists()

    comparison = json.loads(artifacts.comparison.read_text())
    pass_report = json.loads(artifacts.pass_report.read_text())
    optimized_graph = json.loads(artifacts.optimized_graph.read_text())
    optimized_targets = [node["target"] for node in optimized_graph["nodes"]]

    assert comparison["max_abs_diff"] == 0.0
    assert pass_report["graph_meta"]["fused_rmsnorm_swiglu_aten_matches"] == 1
    assert any("fused_rmsnorm_swiglu" in target for target in optimized_targets)

    subprocess.run(
        [
            sys.executable,
            str(artifacts.repro),
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path / "pattern-repro"),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert (tmp_path / "pattern-repro" / "comparison.json").exists()


def test_profile_focus_cli_commands_write_manifests(tmp_path) -> None:
    env = {**os.environ, "PYTHONPATH": "src"}
    region_dir = tmp_path / "cli-region"
    pattern_dir = tmp_path / "cli-pattern"

    region = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchinferno.cli",
            "profile-region",
            str(region_dir),
            "--region",
            "layers.0.moe",
            "--device",
            "cpu",
            "--batch-size",
            "1",
            "--tokens",
            "2",
            "--warmup",
            "0",
            "--iters",
            "1",
            "--no-profiler",
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )
    pattern = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchinferno.cli",
            "profile-pattern",
            str(pattern_dir),
            "--device",
            "cpu",
            "--batch-size",
            "1",
            "--tokens",
            "2",
            "--hidden-size",
            "16",
            "--warmup",
            "0",
            "--iters",
            "1",
            "--no-profiler",
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    region_manifest = json.loads((region_dir / "manifest.json").read_text())
    pattern_manifest = json.loads((pattern_dir / "manifest.json").read_text())
    assert "TorchInferno region profile" in region.stdout
    assert "TorchInferno pattern profile" in pattern.stdout
    assert region_manifest["artifacts"]["operator_profile"] is None
    assert pattern_manifest["artifacts"]["pass_report"] == "pass_report.json"


def test_subgraph_profile_extracts_node_ids_from_source_run(tmp_path) -> None:
    source_dir = tmp_path / "source"
    source = run_profile_capture(
        ProfileRunConfig(
            output_dir=source_dir,
            device="cpu",
            batch_size=1,
            prompt_tokens=2,
            new_tokens=1,
            warmup=0,
            capture_profiler=False,
        )
    )
    graph = json.loads(source.graph_json.read_text())
    embedding_id = next(node["id"] for node in graph["nodes"] if node["target"] == "aten.embedding.default")

    artifacts = run_subgraph_profile_capture(
        SubgraphProfileConfig(
            output_dir=tmp_path / "subgraph",
            source_run_dir=source_dir,
            node_ids=(embedding_id,),
            device="cpu",
            warmup=0,
            iters=1,
        )
    )

    spec = json.loads(artifacts.subgraph_spec.read_text())
    output = json.loads(artifacts.output.read_text())
    subgraph = json.loads(artifacts.subgraph_graph.read_text())

    assert spec["requested_node_ids"] == [embedding_id]
    assert spec["boundary_inputs"][0]["source_op"] == "placeholder"
    assert output["max_abs_diff_vs_source"] == 0.0
    assert any(node["target"] == "aten.embedding.default" for node in subgraph["nodes"])

    subprocess.run(
        [
            sys.executable,
            str(artifacts.repro),
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path / "subgraph-repro"),
            "--source-run",
            str(source_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert (tmp_path / "subgraph-repro" / "subgraph_spec.json").exists()


def test_subgraph_profile_cli_and_node_listing(tmp_path) -> None:
    env = {**os.environ, "PYTHONPATH": "src"}
    source_dir = tmp_path / "cli-source"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torchinferno.cli",
            "profile-run",
            str(source_dir),
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
    nodes = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchinferno.cli",
            "profile-nodes",
            str(source_dir),
            "--grep",
            "embedding",
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )
    embedding_id = int(nodes.stdout.split()[0])
    subgraph_dir = tmp_path / "cli-subgraph"
    subgraph = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchinferno.cli",
            "profile-subgraph",
            str(subgraph_dir),
            "--source-run",
            str(source_dir),
            "--nodes",
            str(embedding_id),
            "--device",
            "cpu",
            "--warmup",
            "0",
            "--iters",
            "1",
            "--no-profiler",
        ],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )

    manifest = json.loads((subgraph_dir / "manifest.json").read_text())
    assert "TorchInferno subgraph profile" in subgraph.stdout
    assert manifest["artifacts"]["operator_profile"] is None
    assert manifest["artifacts"]["subgraph_spec"] == "subgraph_spec.json"
