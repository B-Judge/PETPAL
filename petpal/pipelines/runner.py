"""YAML-driven pipeline runner for PETPAL.

Each step in the YAML file maps to a registered ``petpal-*`` CLI command.
Independent steps are executed in parallel; dependent steps wait for all
their declared dependencies to finish first.

Dependency guarantee
--------------------
Steps are grouped into *execution levels* by Kahn's topological sort.  The
runner processes levels one at a time.  Within each level every step runs in
a :class:`~concurrent.futures.ThreadPoolExecutor`.  The executor's context
manager calls ``shutdown(wait=True)`` when the ``with`` block exits, which
blocks until **every** future in that level has completed.  The next level's
futures are only submitted in the following loop iteration — after the
previous ``with`` block has returned.  Therefore no step in level *N+1* can
start before every step in level *N* has finished.

Example YAML::

    pipeline:
      name: my_pet_analysis
      steps:
        - name: moco
          command: petpal-moco
          args:
            pet: /data/sub-01/pet.nii.gz
            output: /out/moco.nii.gz

        - name: register
          depends_on: [moco]
          command: petpal-register
          args:
            pet: ${moco.args.output}
            reference: /data/sub-01/T1w.nii.gz
            output: /out/reg.nii.gz

Usage::

    petpal-run pipeline.yaml
    petpal-run pipeline.yaml --workers 8 --log-level DEBUG
"""
from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import yaml

from .yaml_schema import Pipeline, PipelineStep

logger = logging.getLogger(__name__)

_REF_PATTERN = re.compile(r'\$\{(\w+)\.args\.(\w+)\}')


def load_pipeline(path: Path | str) -> Pipeline:
    """Load and parse a YAML pipeline file.

    Args:
        path: Path to the YAML pipeline file.

    Returns:
        Parsed :class:`Pipeline` object.

    Raises:
        KeyError: If a step is missing the required ``name`` or ``command`` fields.
    """
    path = Path(path)
    with path.open() as fh:
        data = yaml.safe_load(fh)

    pipeline_data = data.get('pipeline', {})
    name = pipeline_data.get('name', path.stem)
    steps = [
        PipelineStep(
            name=raw['name'],
            command=raw['command'],
            args={str(k): str(v) for k, v in raw.get('args', {}).items()},
            depends_on=list(raw.get('depends_on', [])),
        )
        for raw in pipeline_data.get('steps', [])
    ]
    return Pipeline(name=name, steps=steps)


def resolve_references(pipeline: Pipeline) -> Pipeline:
    """Expand ``${step_name.args.key}`` references in step args.

    References are resolved in declaration order.  A step may only
    reference steps that appear before it in the YAML file.

    Args:
        pipeline: Pipeline with potentially unresolved ``${}`` references.

    Returns:
        New :class:`Pipeline` with all references expanded to literal values.

    Raises:
        ValueError: If a reference names an unknown step or a non-existent argument.
    """
    resolved: dict[str, dict[str, str]] = {}
    new_steps: list[PipelineStep] = []

    for step in pipeline.steps:
        def _make_expander(step_name: str):
            def _expand(match: re.Match) -> str:
                ref_step = match.group(1)
                ref_key = match.group(2)
                if ref_step not in resolved:
                    raise ValueError(
                        f"Step '{step_name}' references '{ref_step}.args.{ref_key}', "
                        f"but '{ref_step}' has not been declared before '{step_name}'."
                    )
                if ref_key not in resolved[ref_step]:
                    raise ValueError(
                        f"Step '{ref_step}' has no argument '{ref_key}'."
                    )
                return resolved[ref_step][ref_key]
            return _expand

        expander = _make_expander(step.name)
        new_args = {k: _REF_PATTERN.sub(expander, v) for k, v in step.args.items()}
        resolved[step.name] = new_args
        new_steps.append(PipelineStep(
            name=step.name,
            command=step.command,
            args=new_args,
            depends_on=step.depends_on,
        ))

    return Pipeline(name=pipeline.name, steps=new_steps)


def _build_execution_levels(pipeline: Pipeline) -> list[list[PipelineStep]]:
    """Group steps into execution levels via Kahn's topological sort.

    All steps within the same level are mutually independent and can run in
    parallel.  Levels must be executed sequentially: every step in level *N*
    must finish before any step in level *N+1* begins.

    Args:
        pipeline: A resolved pipeline (call :func:`resolve_references` first).

    Returns:
        List of levels; each level is a list of steps that can run in parallel.

    Raises:
        ValueError: If ``depends_on`` names an unknown step, or if the
            dependency graph contains a cycle.
    """
    step_map = {s.name: s for s in pipeline.steps}
    in_degree: dict[str, int] = {s.name: len(s.depends_on) for s in pipeline.steps}
    dependents: dict[str, list[str]] = {s.name: [] for s in pipeline.steps}

    for step in pipeline.steps:
        for dep in step.depends_on:
            if dep not in step_map:
                raise ValueError(
                    f"Step '{step.name}' depends on unknown step '{dep}'."
                )
            dependents[dep].append(step.name)

    ready = [name for name, deg in in_degree.items() if deg == 0]
    levels: list[list[PipelineStep]] = []

    while ready:
        levels.append([step_map[name] for name in ready])
        next_ready: list[str] = []
        for name in ready:
            for dep_name in dependents[name]:
                in_degree[dep_name] -= 1
                if in_degree[dep_name] == 0:
                    next_ready.append(dep_name)
        ready = next_ready

    if sum(len(lvl) for lvl in levels) != len(pipeline.steps):
        raise ValueError(
            "Cycle detected in the pipeline dependency graph.  "
            "Check 'depends_on' fields for circular references."
        )

    return levels


def _run_step(step: PipelineStep) -> None:
    """Execute a single pipeline step as a subprocess.

    Translates ``step.command`` and ``step.args`` into a CLI call::

        command --key1 value1 --key2 value2 ...

    Args:
        step: Step to execute.

    Raises:
        subprocess.CalledProcessError: If the command exits with a non-zero
            return code.
    """
    cmd = [step.command]
    for key, value in step.args.items():
        cmd.extend([f"--{key}", value])

    logger.info("Starting  '%s': %s", step.name, " ".join(cmd))
    subprocess.run(cmd, check=True)
    logger.info("Completed '%s'.", step.name)


def run_pipeline(pipeline: Pipeline, max_workers: int = 4) -> None:
    """Execute a resolved pipeline, parallelising independent steps.

    Dependency guarantee
    ~~~~~~~~~~~~~~~~~~~~
    Steps are grouped into execution levels by :func:`_build_execution_levels`.
    Each level is run inside a :class:`~concurrent.futures.ThreadPoolExecutor`
    ``with`` block.  Python's ``ThreadPoolExecutor.__exit__`` calls
    ``shutdown(wait=True)``, which **blocks until every future submitted in
    that level has finished** before the ``with`` block returns.  The next
    level's futures are submitted only in the subsequent loop iteration —
    after the previous ``with`` block has fully exited.  This hard barrier
    guarantees that no step in level *N+1* can start before every step in
    level *N* has completed.

    Fail-fast behaviour
    ~~~~~~~~~~~~~~~~~~~
    After each level's barrier, all completed futures are inspected.  If any
    step failed, every failure is logged and the pipeline exits with code 1
    without submitting any further levels.

    Args:
        pipeline: Resolved pipeline (call :func:`resolve_references` first).
        max_workers: Maximum parallel worker threads per level.

    Raises:
        SystemExit: With code 1 if one or more steps fail.
    """
    levels = _build_execution_levels(pipeline)
    n_levels = len(levels)

    for idx, level in enumerate(levels, start=1):
        names = [s.name for s in level]
        logger.info("Level %d/%d — submitting: %s", idx, n_levels, names)

        # ------------------------------------------------------------------
        # BARRIER
        # Every future for this level is submitted to the pool inside the
        # `with` block.  When the block exits, ThreadPoolExecutor calls
        # shutdown(wait=True), which blocks here until ALL submitted futures
        # have finished — whether they succeeded or raised an exception.
        # The next level's steps are only submitted after this line returns.
        # ------------------------------------------------------------------
        step_futures: dict[Future, PipelineStep] = {}
        with ThreadPoolExecutor(max_workers=min(max_workers, len(level))) as pool:
            step_futures = {pool.submit(_run_step, step): step for step in level}
        # All level-N futures are guaranteed done at this point.

        failures = [
            (step, exc)
            for future, step in step_futures.items()
            if (exc := future.exception()) is not None
        ]
        if failures:
            for step, exc in failures:
                logger.error("Step '%s' failed: %s", step.name, exc)
            logger.error(
                "%d step(s) failed at level %d/%d — downstream steps will not run.",
                len(failures), idx, n_levels,
            )
            sys.exit(1)

        logger.info("Level %d/%d — all steps completed.", idx, n_levels)


def main() -> None:
    """Entry point for the ``petpal-run`` command."""
    parser = argparse.ArgumentParser(
        prog="petpal-run",
        description="Run a PETPAL pipeline defined in a YAML file.",
    )
    parser.add_argument(
        "pipeline",
        metavar="PIPELINE_YAML",
        type=Path,
        help="Path to the YAML pipeline file.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Maximum parallel worker threads per execution level (default: 4).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    pipeline = load_pipeline(args.pipeline)
    pipeline = resolve_references(pipeline)
    run_pipeline(pipeline, max_workers=args.workers)
