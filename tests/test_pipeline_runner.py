"""Tests for the PETPAL YAML pipeline runner."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from petpal.pipelines.runner import (
    _build_execution_levels,
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
    """A minimal two-step serial pipeline."""
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
    """A pipeline whose two leaf steps can run in parallel."""
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
# run_pipeline (subprocess mocked)
# ---------------------------------------------------------------------------


def test_run_pipeline_invokes_all_steps(serial_yaml: Path, monkeypatch) -> None:
    calls: list[str] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd[0])

    monkeypatch.setattr("petpal.pipelines.runner.subprocess.run", fake_run)
    pipeline = resolve_references(load_pipeline(serial_yaml))
    run_pipeline(pipeline, max_workers=2)

    assert set(calls) == {"petpal-moco", "petpal-register"}


def test_run_pipeline_fails_fast(serial_yaml: Path, monkeypatch) -> None:
    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd[0])

    monkeypatch.setattr("petpal.pipelines.runner.subprocess.run", fake_run)
    pipeline = resolve_references(load_pipeline(serial_yaml))

    with pytest.raises(SystemExit) as exc:
        run_pipeline(pipeline, max_workers=1)
    assert exc.value.code == 1
