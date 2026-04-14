"""Tests for the PETPAL YAML pipeline runner.

Coverage
--------
* load_pipeline   — YAML parsing and field mapping
* resolve_references — ``${step.args.key}`` expansion
* _build_execution_levels — Kahn topological sort, cycle/unknown-dep errors
* run_pipeline — subprocess dispatch, **strict dependency ordering**,
  fail-fast behaviour, downstream isolation on failure
* _run_step — CLI flag translation (``{"key": "val"}`` → ``["--key", "val"]``)
"""
from __future__ import annotations

import subprocess
import threading
from pathlib import Path

import pytest
import yaml

from petpal.pipelines.runner import (
    _build_execution_levels,
    _run_step,
    load_pipeline,
    resolve_references,
    run_pipeline,
)
from petpal.pipelines.yaml_schema import Pipeline, PipelineStep


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def serial_yaml(tmp_path: Path) -> Path:
    """A minimal two-step serial pipeline: moco → register."""
    content = {
        "pipeline": {
            "name": "serial_pipeline",
            "steps": [
                {
                    "name": "moco",
                    "command": "petpal-moco",
                    "args": {
                        "pet": "/in/pet.nii.gz",
                        "output": "/out/moco.nii.gz",
                    },
                },
                {
                    "name": "register",
                    "command": "petpal-register",
                    "depends_on": ["moco"],
                    "args": {
                        "pet": "${moco.args.output}",
                        "reference": "/data/t1.nii.gz",
                        "output": "/out/reg.nii.gz",
                    },
                },
            ],
        }
    }
    path = tmp_path / "serial.yaml"
    path.write_text(yaml.dump(content))
    return path


@pytest.fixture
def parallel_yaml(tmp_path: Path) -> Path:
    """Pipeline where two leaf steps share one upstream dependency.

    extract_tacs → patlak
                 → logan
    """
    content = {
        "pipeline": {
            "name": "parallel_pipeline",
            "steps": [
                {
                    "name": "extract_tacs",
                    "command": "petpal-write-tacs",
                    "args": {
                        "pet": "/out/reg.nii.gz",
                        "output": "/out/tacs.tsv",
                    },
                },
                {
                    "name": "patlak",
                    "command": "petpal-graph-analysis",
                    "depends_on": ["extract_tacs"],
                    "args": {
                        "tac": "${extract_tacs.args.output}",
                        "output": "/out/patlak.nii.gz",
                    },
                },
                {
                    "name": "logan",
                    "command": "petpal-graph-analysis",
                    "depends_on": ["extract_tacs"],
                    "args": {
                        "tac": "${extract_tacs.args.output}",
                        "output": "/out/logan.nii.gz",
                    },
                },
            ],
        }
    }
    path = tmp_path / "parallel.yaml"
    path.write_text(yaml.dump(content))
    return path


@pytest.fixture
def diamond_yaml(tmp_path: Path) -> Path:
    """Diamond dependency graph: prep → [branch_a, branch_b] → merge.

    branch_a and branch_b run in parallel (level 1).
    merge waits for *both* (level 2).
    """
    content = {
        "pipeline": {
            "name": "diamond_pipeline",
            "steps": [
                {
                    "name": "prep",
                    "command": "petpal-prep",
                    "args": {"input": "/in/raw.nii.gz", "output": "/out/prep.nii.gz"},
                },
                {
                    "name": "branch_a",
                    "command": "petpal-branch-a",
                    "depends_on": ["prep"],
                    "args": {"input": "${prep.args.output}", "output": "/out/a.nii.gz"},
                },
                {
                    "name": "branch_b",
                    "command": "petpal-branch-b",
                    "depends_on": ["prep"],
                    "args": {"input": "${prep.args.output}", "output": "/out/b.nii.gz"},
                },
                {
                    "name": "merge",
                    "command": "petpal-merge",
                    "depends_on": ["branch_a", "branch_b"],
                    "args": {
                        "input_a": "${branch_a.args.output}",
                        "input_b": "${branch_b.args.output}",
                        "output": "/out/merged.nii.gz",
                    },
                },
            ],
        }
    }
    path = tmp_path / "diamond.yaml"
    path.write_text(yaml.dump(content))
    return path


@pytest.fixture
def chain_yaml(tmp_path: Path) -> Path:
    """Strict three-level serial chain: step_a → step_b → step_c."""
    content = {
        "pipeline": {
            "name": "chain_pipeline",
            "steps": [
                {
                    "name": "step_a",
                    "command": "petpal-a",
                    "args": {"output": "/out/a.nii.gz"},
                },
                {
                    "name": "step_b",
                    "command": "petpal-b",
                    "depends_on": ["step_a"],
                    "args": {"input": "${step_a.args.output}", "output": "/out/b.nii.gz"},
                },
                {
                    "name": "step_c",
                    "command": "petpal-c",
                    "depends_on": ["step_b"],
                    "args": {"input": "${step_b.args.output}", "output": "/out/c.nii.gz"},
                },
            ],
        }
    }
    path = tmp_path / "chain.yaml"
    path.write_text(yaml.dump(content))
    return path


# ---------------------------------------------------------------------------
# load_pipeline
# ---------------------------------------------------------------------------


def test_load_pipeline_name(serial_yaml: Path) -> None:
    assert load_pipeline(serial_yaml).name == "serial_pipeline"


def test_load_pipeline_step_count(serial_yaml: Path) -> None:
    assert len(load_pipeline(serial_yaml).steps) == 2


def test_load_pipeline_step_fields(serial_yaml: Path) -> None:
    steps = load_pipeline(serial_yaml).steps
    assert steps[0].name == "moco"
    assert steps[0].command == "petpal-moco"
    assert steps[1].depends_on == ["moco"]


def test_load_pipeline_step_args(serial_yaml: Path) -> None:
    args = load_pipeline(serial_yaml).steps[0].args
    assert args["pet"] == "/in/pet.nii.gz"
    assert args["output"] == "/out/moco.nii.gz"


# ---------------------------------------------------------------------------
# resolve_references
# ---------------------------------------------------------------------------


def test_resolve_expands_reference(serial_yaml: Path) -> None:
    resolved = resolve_references(load_pipeline(serial_yaml))
    assert resolved.steps[1].args["pet"] == "/out/moco.nii.gz"


def test_resolve_leaves_literals_unchanged(serial_yaml: Path) -> None:
    resolved = resolve_references(load_pipeline(serial_yaml))
    assert resolved.steps[1].args["reference"] == "/data/t1.nii.gz"


def test_resolve_parallel_both_expanded(parallel_yaml: Path) -> None:
    resolved = resolve_references(load_pipeline(parallel_yaml))
    assert resolved.steps[1].args["tac"] == "/out/tacs.tsv"
    assert resolved.steps[2].args["tac"] == "/out/tacs.tsv"


def test_resolve_unknown_step_raises() -> None:
    pipeline = Pipeline(
        name="bad",
        steps=[
            PipelineStep(
                name="step_b",
                command="cmd",
                args={"x": "${missing.args.output}"},
            )
        ],
    )
    with pytest.raises(ValueError, match="missing"):
        resolve_references(pipeline)


def test_resolve_unknown_arg_raises() -> None:
    pipeline = Pipeline(
        name="bad",
        steps=[
            PipelineStep("a", "cmd", {"out": "/tmp/a.nii.gz"}),
            PipelineStep("b", "cmd", {"x": "${a.args.nonexistent}"}, depends_on=["a"]),
        ],
    )
    with pytest.raises(ValueError, match="nonexistent"):
        resolve_references(pipeline)


# ---------------------------------------------------------------------------
# _build_execution_levels
# ---------------------------------------------------------------------------


def test_levels_serial(serial_yaml: Path) -> None:
    pipeline = resolve_references(load_pipeline(serial_yaml))
    levels = _build_execution_levels(pipeline)
    assert len(levels) == 2
    assert levels[0][0].name == "moco"
    assert levels[1][0].name == "register"


def test_levels_parallel_leaf_steps(parallel_yaml: Path) -> None:
    pipeline = resolve_references(load_pipeline(parallel_yaml))
    levels = _build_execution_levels(pipeline)
    assert len(levels) == 2
    assert levels[0][0].name == "extract_tacs"
    assert {s.name for s in levels[1]} == {"patlak", "logan"}


def test_levels_diamond(diamond_yaml: Path) -> None:
    """Diamond graph must produce exactly 3 levels."""
    pipeline = resolve_references(load_pipeline(diamond_yaml))
    levels = _build_execution_levels(pipeline)
    assert len(levels) == 3
    assert levels[0][0].name == "prep"
    assert {s.name for s in levels[1]} == {"branch_a", "branch_b"}
    assert levels[2][0].name == "merge"


def test_levels_chain(chain_yaml: Path) -> None:
    """Three-step serial chain must produce three single-step levels."""
    pipeline = resolve_references(load_pipeline(chain_yaml))
    levels = _build_execution_levels(pipeline)
    assert len(levels) == 3
    assert [lvl[0].name for lvl in levels] == ["step_a", "step_b", "step_c"]


def test_levels_cycle_raises() -> None:
    pipeline = Pipeline(
        name="cyclic",
        steps=[
            PipelineStep("a", "cmd", {}, depends_on=["b"]),
            PipelineStep("b", "cmd", {}, depends_on=["a"]),
        ],
    )
    with pytest.raises(ValueError, match="[Cc]ycle"):
        _build_execution_levels(pipeline)


def test_levels_unknown_dep_raises() -> None:
    pipeline = Pipeline(
        name="bad",
        steps=[PipelineStep("a", "cmd", {}, depends_on=["ghost"])],
    )
    with pytest.raises(ValueError, match="ghost"):
        _build_execution_levels(pipeline)


# ---------------------------------------------------------------------------
# run_pipeline — subprocess mocked
# ---------------------------------------------------------------------------


def test_run_pipeline_invokes_all_steps(serial_yaml: Path, monkeypatch) -> None:
    """Both steps are called when the pipeline succeeds."""
    calls: list[str] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd[0])

    monkeypatch.setattr("petpal.pipelines.runner.subprocess.run", fake_run)
    pipeline = resolve_references(load_pipeline(serial_yaml))
    run_pipeline(pipeline, max_workers=2)

    assert set(calls) == {"petpal-moco", "petpal-register"}


def test_run_pipeline_serial_order(serial_yaml: Path, monkeypatch) -> None:
    """moco must be called *before* register in a serial pipeline.

    This verifies the dependency barrier: register lives in level 1 and
    must not start until level 0 (moco) has fully completed.
    """
    call_order: list[str] = []
    lock = threading.Lock()

    def fake_run(cmd, **kwargs):
        with lock:
            call_order.append(cmd[0])

    monkeypatch.setattr("petpal.pipelines.runner.subprocess.run", fake_run)
    pipeline = resolve_references(load_pipeline(serial_yaml))
    run_pipeline(pipeline, max_workers=4)

    assert call_order.index("petpal-moco") < call_order.index("petpal-register"), (
        "register ran before moco — dependency barrier violated"
    )


def test_run_pipeline_diamond_order(diamond_yaml: Path, monkeypatch) -> None:
    """prep must finish before both branches; both branches must finish before merge.

    Verifies three-level dependency ordering in a diamond graph.
    """
    call_order: list[str] = []
    lock = threading.Lock()

    def fake_run(cmd, **kwargs):
        with lock:
            call_order.append(cmd[0])

    monkeypatch.setattr("petpal.pipelines.runner.subprocess.run", fake_run)
    pipeline = resolve_references(load_pipeline(diamond_yaml))
    run_pipeline(pipeline, max_workers=4)

    prep_idx = call_order.index("petpal-prep")
    merge_idx = call_order.index("petpal-merge")
    branch_a_idx = call_order.index("petpal-branch-a")
    branch_b_idx = call_order.index("petpal-branch-b")

    assert prep_idx < branch_a_idx, "branch_a started before prep finished"
    assert prep_idx < branch_b_idx, "branch_b started before prep finished"
    assert branch_a_idx < merge_idx, "merge started before branch_a finished"
    assert branch_b_idx < merge_idx, "merge started before branch_b finished"


def test_run_pipeline_three_level_chain(chain_yaml: Path, monkeypatch) -> None:
    """step_a → step_b → step_c must execute in strict order."""
    call_order: list[str] = []
    lock = threading.Lock()

    def fake_run(cmd, **kwargs):
        with lock:
            call_order.append(cmd[0])

    monkeypatch.setattr("petpal.pipelines.runner.subprocess.run", fake_run)
    pipeline = resolve_references(load_pipeline(chain_yaml))
    run_pipeline(pipeline, max_workers=4)

    assert call_order == ["petpal-a", "petpal-b", "petpal-c"], (
        f"Expected strict serial order, got: {call_order}"
    )


def test_run_pipeline_fails_fast(serial_yaml: Path, monkeypatch) -> None:
    """A failing step causes sys.exit(1)."""
    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd[0])

    monkeypatch.setattr("petpal.pipelines.runner.subprocess.run", fake_run)
    pipeline = resolve_references(load_pipeline(serial_yaml))

    with pytest.raises(SystemExit) as exc:
        run_pipeline(pipeline, max_workers=1)
    assert exc.value.code == 1


def test_downstream_not_called_on_failure(serial_yaml: Path, monkeypatch) -> None:
    """When moco fails, register must never be invoked.

    This is the key safety guarantee: no downstream step may run when an
    upstream dependency has failed.
    """
    called: list[str] = []

    def fake_run(cmd, **kwargs):
        called.append(cmd[0])
        if cmd[0] == "petpal-moco":
            raise subprocess.CalledProcessError(1, cmd[0])

    monkeypatch.setattr("petpal.pipelines.runner.subprocess.run", fake_run)
    pipeline = resolve_references(load_pipeline(serial_yaml))

    with pytest.raises(SystemExit):
        run_pipeline(pipeline, max_workers=1)

    assert "petpal-moco" in called, "moco should have been attempted"
    assert "petpal-register" not in called, (
        "register was called despite moco failing — downstream isolation violated"
    )


# ---------------------------------------------------------------------------
# _run_step — CLI flag translation
# ---------------------------------------------------------------------------


def test_run_step_translates_args_to_flags(monkeypatch) -> None:
    """Step args dict must be translated to --key value pairs on the CLI."""
    captured: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured.append(cmd)

    monkeypatch.setattr("petpal.pipelines.runner.subprocess.run", fake_run)

    step = PipelineStep(
        name="moco",
        command="petpal-moco",
        args={"pet": "/in/pet.nii.gz", "output": "/out/moco.nii.gz"},
    )
    _run_step(step)

    assert len(captured) == 1
    cmd = captured[0]
    assert cmd[0] == "petpal-moco"
    assert "--pet" in cmd
    assert cmd[cmd.index("--pet") + 1] == "/in/pet.nii.gz"
    assert "--output" in cmd
    assert cmd[cmd.index("--output") + 1] == "/out/moco.nii.gz"


def test_run_step_no_args_yields_bare_command(monkeypatch) -> None:
    """A step with no args produces a command list of length 1."""
    captured: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        captured.append(cmd)

    monkeypatch.setattr("petpal.pipelines.runner.subprocess.run", fake_run)

    step = PipelineStep(name="noop", command="petpal-noop", args={})
    _run_step(step)

    assert captured[0] == ["petpal-noop"]
