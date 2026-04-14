"""PETPAL pipeline runner.

Two primitives replace the old DAG pipeline system:

1. **Per-function CLIs** — every scientific function is independently
   callable from the command line (see ``petpal/cli/``).
2. **YAML pipeline files** — declarative step definitions that are parsed
   and executed with automatic parallelisation of independent steps.

Quick start::

    petpal-run my_pipeline.yaml

Programmatic use::

    from petpal.pipelines import load_pipeline, resolve_references, run_pipeline

    pipeline = load_pipeline("my_pipeline.yaml")
    pipeline = resolve_references(pipeline)
    run_pipeline(pipeline, max_workers=8)
"""

from .runner import load_pipeline, resolve_references, run_pipeline
from .yaml_schema import Pipeline, PipelineStep

__all__ = [
    "Pipeline",
    "PipelineStep",
    "load_pipeline",
    "resolve_references",
    "run_pipeline",
]
