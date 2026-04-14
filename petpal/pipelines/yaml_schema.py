"""Dataclasses defining the PETPAL YAML pipeline schema."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PipelineStep:
    """A single step in a PETPAL pipeline.

    Args:
        name: Unique identifier for this step.
        command: The CLI command to run (e.g. ``petpal-moco``).
        args: Mapping of argument names to values.  Values may use
            ``${step_name.args.key}`` syntax to reference an output
            from a previous step.
        depends_on: Names of steps that must complete before this step
            runs.  Steps with no dependencies run as soon as the pipeline
            starts.
    """

    name: str
    command: str
    args: dict[str, str] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)


@dataclass
class Pipeline:
    """A complete PETPAL pipeline loaded from a YAML file.

    Args:
        name: Human-readable pipeline name.
        steps: Ordered list of pipeline steps.
    """

    name: str
    steps: list[PipelineStep]
