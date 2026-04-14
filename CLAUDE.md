# PETPAL Refactoring Guide

## Project Overview

PETPAL is a scientific Python package (v0.6.0, pre-alpha) for PET/MR medical imaging analysis,
covering kinetic modeling, preprocessing, image-derived input function (IDIF) extraction, and
parametric imaging.

**Tech stack:** Python 3.12+, hatchling build, pytest, Sphinx docs, NumPy/SciPy/nibabel/antspyx/lmfit

---

## Refactoring Goals and Philosophy

**Primary goal:** Simplicity. Dismantle the existing custom DAG pipeline abstraction and replace
it with two flat, composable primitives:

1. **Per-function CLIs** — every meaningful scientific function has its own registered CLI
   command, making the entire library trivially bash-scriptable
2. **YAML pipeline files** — declarative step definitions that are parsed and executed with
   automatic parallelization of independent steps

**Secondary goal:** Code quality — consistent naming, full type hints, Google-style docstrings on
all public interfaces.

**Backward compatibility:** None required. This is pre-alpha; break APIs freely.

---

## Coding Standards

Apply these standards to all code you write or significantly modify:

- **Type hints:** Full annotations on all function signatures and class attributes
- **Docstrings:** Google-style on all public functions, methods, and classes
- **Path arguments:** Use `pathlib.Path | str`; prefer `pathlib.Path` internally
- **Config/parameter objects:** Use `@dataclass` or `@dataclass(frozen=True)` instead of
  untyped dicts
- **File and function size:** Keep reasonable as a guideline (~500 lines/file,
  ~50 lines/function); split when it genuinely aids clarity, not as a hard rule

---

## The Core Architectural Shift

### What to Dismantle

The entire `petpal/pipelines/` directory currently implements a custom DAG-based orchestration
system. **Delete or gut all of the following** — this abstraction is being replaced:

| File | Action |
|------|--------|
| `petpal/pipelines/steps_base.py` | Delete — `StepsAPI`, `FunctionBasedStep`, `ObjectBasedStep` replaced by CLIs |
| `petpal/pipelines/steps_containers.py` | Delete — `StepsContainer`, `StepsPipeline` replaced by YAML runner |
| `petpal/pipelines/pipelines.py` | Delete — `BIDS_Pipeline` replaced by YAML runner |
| `petpal/pipelines/preproc_steps.py` | Delete — each step becomes a standalone CLI |
| `petpal/pipelines/kinetic_modeling_steps.py` | Delete — each step becomes a standalone CLI |
| `petpal/pipelines/pca_guided_idif_steps.py` | Delete — each step becomes a standalone CLI |

### What to Build Instead

**`petpal/pipelines/` becomes three files only:**

```
petpal/pipelines/
├── __init__.py          # minimal; exports run_pipeline
├── runner.py            # YAML pipeline parser + executor
└── yaml_schema.py       # dataclasses defining the YAML schema
```

---

## Primitive 1: Per-Function CLIs

Every meaningful scientific function should be independently callable from the command line.
CLIs live in `petpal/cli/` and are registered as entry points in `pyproject.toml`.

### Design Rules for CLIs

- Accept all inputs as named arguments (`--pet`, `--output`, `--method`, etc.)
- Write outputs to caller-specified paths
- No pipeline framework required to run — each CLI is fully self-contained
- Return exit code 0 on success, non-zero on failure
- Use `argparse` (match existing CLI style in `petpal/cli/`)
- Each CLI module has a `main()` function that is the registered entry point

### Example

```bash
petpal-register --pet sub-01_pet.nii.gz --t1 sub-01_T1w.nii.gz --output reg.nii.gz
petpal-moco --pet sub-01_pet.nii.gz --output moco.nii.gz
petpal-graph-analysis --method patlak --tac tacs.tsv --output ki_map.nii.gz
```

### CLI Coverage Targets

Audit existing CLIs in `petpal/cli/` and expand so every major function has coverage.
Priority targets (functions without adequate CLIs today):

- `petpal/preproc/motion_corr.py` — motion correction
- `petpal/preproc/register.py` — PET-to-T1 registration
- `petpal/preproc/segmentation_tools.py` — brain segmentation
- `petpal/preproc/regional_tac_extraction.py` — TAC extraction from ROIs
- `petpal/preproc/decay_correction.py` — decay correction
- `petpal/preproc/standard_uptake_value.py` — SUV calculation
- `petpal/kinetic_modeling/tac_fitting.py` — TCM fitting
- `petpal/kinetic_modeling/graphical_analysis.py` — Patlak / Logan analysis
- `petpal/kinetic_modeling/parametric_images.py` — voxel-wise parametric maps
- `petpal/input_function/pca_guided_idif.py` — PCA-guided IDIF

Register all new entry points in `pyproject.toml` under `[project.scripts]`.

---

## Primitive 2: YAML Pipeline Runner

A single thin module (`petpal/pipelines/runner.py`) replaces the entire old pipeline system.

### YAML Schema

```yaml
pipeline:
  name: my_pet_analysis

  steps:
    - name: motion_correction
      command: petpal-moco
      args:
        pet: /data/sub-01/pet.nii.gz
        output: /out/moco.nii.gz

    - name: register
      depends_on: [motion_correction]
      command: petpal-register
      args:
        pet: ${motion_correction.args.output}
        t1: /data/sub-01/T1w.nii.gz
        output: /out/reg.nii.gz

    - name: extract_tacs
      depends_on: [register]
      command: petpal-extract-tacs
      args:
        pet: ${register.args.output}
        seg: /data/sub-01/seg.nii.gz
        output: /out/tacs.tsv

    - name: patlak
      depends_on: [extract_tacs]
      command: petpal-graph-analysis
      args:
        method: patlak
        tac: ${extract_tacs.args.output}
        output: /out/patlak.nii.gz

    - name: logan
      depends_on: [extract_tacs]   # parallel with patlak
      command: petpal-graph-analysis
      args:
        method: logan
        tac: ${extract_tacs.args.output}
        output: /out/logan.nii.gz
```

### Runner Behavior

- **`${step_name.args.key}`** — reference another step's argument (for output chaining)
- **`depends_on`** — explicit dependency list; steps without it run as soon as their deps complete
- Steps are dispatched as subprocesses (the registered CLI commands)
- Topological sort determines execution order
- Steps with no unmet dependencies run in parallel via `concurrent.futures.ThreadPoolExecutor`
- On any step failure: log the error and fail fast (do not continue downstream steps)
- Single CLI entry point: `petpal-run pipeline.yaml`

### Schema Dataclasses (`yaml_schema.py`)

```python
@dataclass
class PipelineStep:
    name: str
    command: str
    args: dict[str, str]
    depends_on: list[str] = field(default_factory=list)

@dataclass
class Pipeline:
    name: str
    steps: list[PipelineStep]
```

### Runner Implementation Sketch

```python
# runner.py
def load_pipeline(path: Path | str) -> Pipeline: ...
def resolve_references(pipeline: Pipeline) -> Pipeline: ...   # expand ${...} syntax
def run_pipeline(pipeline: Pipeline, max_workers: int = 4) -> None: ...
def main() -> None: ...   # argparse entry point for petpal-run
```

---

## What NOT to Do

- **Do not** create a new pipeline abstraction class to replace the old one — the YAML runner
  should be as thin as possible
- **Do not** refactor the core scientific algorithm modules (`tac_fitting.py`,
  `graphical_analysis.py`, `parametric_images.py`, etc.) — only add/improve their CLI wrappers
- **Do not** add compatibility shims or deprecation warnings for the old pipeline API
- **Do not** add new scientific features — this refactoring is about structure, not scope
- **Do not** add error handling for impossible cases; trust Python's type system and tests

---

## Testing

- **YAML runner:** Test that `load_pipeline` parses correctly, `resolve_references` expands
  `${...}` syntax, and parallel steps are submitted concurrently
- **CLI modules:** Each new CLI should have a smoke test (invokes `main()` on minimal test data
  without error)
- **Test location:** `tests/` directory; use fixtures from `petpal/utils/testing_utils.py`
- **Framework:** pytest

---

## File Reference

| Path | Purpose |
|------|---------|
| `petpal/cli/` | All CLI entry point modules |
| `pyproject.toml` | CLI registrations under `[project.scripts]` |
| `petpal/pipelines/runner.py` | New YAML pipeline runner (replace entire old pipelines dir) |
| `petpal/pipelines/yaml_schema.py` | Schema dataclasses |
| `petpal/preproc/*.py` | Scientific functions to wrap with CLIs |
| `petpal/kinetic_modeling/*.py` | Scientific functions to wrap with CLIs |
| `petpal/input_function/*.py` | Scientific functions to wrap with CLIs |
| `petpal/utils/bids_utils.py` | BIDS path utilities (shared across CLI modules) |
| `petpal/utils/testing_utils.py` | Shared test fixtures |
