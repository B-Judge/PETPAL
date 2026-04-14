.. _yaml-pipeline-tutorial:

====================================
YAML Pipeline Runner
====================================

PETPAL's pipeline runner lets you describe a complete PET/MR analysis as a
declarative YAML file and run every step — in the right order, with automatic
parallelisation — with a single command::

    petpal-run my_pipeline.yaml

This tutorial covers:

1. :ref:`pipeline-yaml-schema` — how to write a pipeline file
2. :ref:`pipeline-reference-syntax` — wiring step outputs to later inputs
3. :ref:`pipeline-execution-model` — ordering guarantees and parallelism
4. :ref:`pipeline-cli-reference` — all ``petpal-*`` commands at a glance
5. :ref:`pipeline-full-example` — a complete PET/MR walkthrough


.. _pipeline-yaml-schema:

YAML Schema
-----------

Every pipeline file must have a single top-level ``pipeline`` key:

.. code-block:: yaml

    pipeline:
      name: my_analysis          # human-readable label (no spaces)
      steps:
        - name: <step-name>      # unique identifier within the pipeline
          command: petpal-<cmd>  # any registered petpal-* CLI entry point
          depends_on: []         # list of step names this step waits for
          args:                  # key/value pairs passed as --key value flags
            key: value

``depends_on`` is optional; omitting it (or leaving it empty) means the step
has no dependencies and is eligible to run as soon as the runner starts.

.. list-table:: Pipeline YAML fields
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Required
     - Description
   * - ``pipeline.name``
     - yes
     - Identifier for the pipeline, used in log messages.
   * - ``steps[].name``
     - yes
     - Unique step identifier. Used in ``depends_on`` lists and
       ``${...}`` references.
   * - ``steps[].command``
     - yes
     - Executable to run, e.g. ``petpal-moco``.  Must be on ``$PATH``.
   * - ``steps[].depends_on``
     - no
     - List of step names that must complete before this step starts.
       Steps without this field (or with an empty list) are independent
       and may run in parallel with other independent steps.
   * - ``steps[].args``
     - no
     - Mapping of argument names to values.  Each pair is passed to the
       command as ``--key value``.  Values may contain ``${...}``
       references (see :ref:`pipeline-reference-syntax`).


.. _pipeline-reference-syntax:

Reference Syntax — ``${step.args.key}``
-----------------------------------------

Step arguments often need to use the *output* of an earlier step as their
*input*.  Instead of repeating the path in two places, use a reference:

.. code-block:: yaml

    - name: moco
      command: petpal-moco
      args:
        pet: /data/sub-01/pet.nii.gz
        output: /out/moco.nii.gz       # ← declared here

    - name: register
      command: petpal-register
      depends_on: [moco]
      args:
        pet: ${moco.args.output}       # ← resolved to /out/moco.nii.gz
        reference: /data/sub-01/T1w.nii.gz
        output: /out/reg.nii.gz

The syntax is::

    ${<step-name>.args.<arg-key>}

The runner substitutes the reference with the literal string value of
``<step-name>.args.<arg-key>`` before any subprocess is launched.  References
to unknown steps or unknown argument keys raise a ``ValueError`` at load time,
before any work is done.

.. note::

   A step may only reference steps declared *before* it in the YAML file.
   Forward references are not supported.


.. _pipeline-execution-model:

Execution Model
---------------

Dependency ordering
~~~~~~~~~~~~~~~~~~~

The runner uses **Kahn's topological sort** to group all steps into
*execution levels*:

* **Level 0** — steps with no ``depends_on`` (roots)
* **Level 1** — steps whose every dependency is in level 0
* **Level N** — steps whose every dependency is in a level < N

Levels are executed strictly one at a time.  All steps within a level are
submitted to a :class:`~concurrent.futures.ThreadPoolExecutor` and the pool's
context manager calls ``shutdown(wait=True)`` before the next level begins.
This hard barrier guarantees:

.. important::

   **No step in level N+1 can start before every step in level N has
   finished**, regardless of how many worker threads are used.

Parallelism
~~~~~~~~~~~

Steps within the same level that share no dependencies run in parallel.  The
number of concurrent threads is controlled by ``--workers`` (default: 4):

.. code-block:: bash

    petpal-run pipeline.yaml --workers 8

Fail-fast behaviour
~~~~~~~~~~~~~~~~~~~

If one or more steps in a level fail, the runner:

1. Waits for every other step in that level to finish (or fail).
2. Logs all failures.
3. Exits with code ``1`` **without submitting any further levels**.

This ensures downstream steps are never called with missing or corrupt inputs.

Example — execution order for the full PET pipeline::

    Level 0:  moco
    Level 1:  register              (depends_on: moco)
    Level 2:  resample_seg          (depends_on: register)
    Level 3:  write_tacs            (depends_on: register, resample_seg)
    Level 4:  patlak  ─┐            (both depend_on: write_tacs only)
              logan   ─┘            ← these two run in parallel


.. _pipeline-cli-reference:

CLI Reference
-------------

All commands are registered in ``pyproject.toml`` and available after
installing PETPAL (``pip install -e .``).

``petpal-run``
~~~~~~~~~~~~~~

Run a YAML pipeline::

    petpal-run PIPELINE_YAML [--workers N] [--log-level LEVEL]

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Description
   * - ``PIPELINE_YAML``
     - Path to the ``.yaml`` pipeline file (positional).
   * - ``--workers N``
     - Maximum parallel threads per execution level (default: 4).
   * - ``--log-level LEVEL``
     - Verbosity: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR`` (default: ``INFO``).

``petpal-moco``
~~~~~~~~~~~~~~~

Motion-correct a 4-D PET series.

.. code-block:: bash

    petpal-moco windowed --pet PET --output OUT [--window-length N] [--step-size N]
    petpal-moco frames-above-mean --pet PET --output OUT

``petpal-register``
~~~~~~~~~~~~~~~~~~~

Register PET to anatomical or atlas space.

.. code-block:: bash

    petpal-register to-t1      --pet PET --reference T1  --output OUT
    petpal-register to-atlas   --pet PET --anat T1 --atlas ATLAS --output OUT
    petpal-register apply-xfm  --pet PET --reference REF --xfm XFM --output OUT

``petpal-seg``
~~~~~~~~~~~~~~

Segmentation utilities.

.. code-block:: bash

    petpal-seg resample  --seg SEG --reference REF --output OUT
    petpal-seg erode-wm  --seg SEG --output OUT
    petpal-seg vat-wm    --seg SEG --output OUT

``petpal-write-tacs``
~~~~~~~~~~~~~~~~~~~~~

Extract regional time-activity curves.

.. code-block:: bash

    petpal-write-tacs --pet PET --seg SEG --output TACS_TSV

``petpal-decay-correct``
~~~~~~~~~~~~~~~~~~~~~~~~

Apply or undo radiotracer decay correction.

.. code-block:: bash

    petpal-decay-correct apply --pet PET --half-life HL --scan-time T --output OUT
    petpal-decay-correct undo  --pet PET --half-life HL --scan-time T --output OUT

``petpal-suv``
~~~~~~~~~~~~~~

Compute standardised uptake values.

.. code-block:: bash

    petpal-suv suv          --pet PET --weight KG --dose MBQ --output OUT
    petpal-suv suvr         --pet PET --reference-tac REF --output OUT
    petpal-suv weighted-sum --pet PET --output OUT

``petpal-pca-idif``
~~~~~~~~~~~~~~~~~~~

Image-derived input function via PCA.

.. code-block:: bash

    petpal-pca-idif top-voxels --pet PET --output OUT
    petpal-pca-idif fitter     --pet PET --output OUT

``petpal-graph-analysis``
~~~~~~~~~~~~~~~~~~~~~~~~~

Patlak and Logan graphical analysis (entry point registered separately in
``pyproject.toml`` after implementing the CLI wrapper).

.. code-block:: bash

    petpal-graph-analysis --method patlak --tac TACS --output OUT [--t-star T]
    petpal-graph-analysis --method logan  --tac TACS --output OUT [--t-star T]


.. _pipeline-full-example:

Full Example — FDG PET/MR Analysis
-----------------------------------

The file ``examples/full_pet_pipeline.yaml`` (in the PETPAL repository root)
shows a complete workflow for a single subject.  Here is an annotated excerpt:

.. code-block:: yaml

    pipeline:
      name: full_pet_analysis
      steps:

        # Level 0 — no upstream dependencies
        - name: moco
          command: petpal-moco
          args:
            subcommand: windowed
            pet: /data/sub-01/pet/sub-01_trc-FDG_pet.nii.gz
            output: /out/sub-01/moco.nii.gz

        # Level 1 — waits for moco
        - name: register
          command: petpal-register
          depends_on: [moco]
          args:
            subcommand: to-t1
            pet: ${moco.args.output}          # resolved at runtime
            reference: /data/sub-01/anat/sub-01_T1w.nii.gz
            output: /out/sub-01/reg.nii.gz

        # Level 2 — waits for register
        - name: resample_seg
          command: petpal-seg
          depends_on: [register]
          args:
            subcommand: resample
            seg: /data/sub-01/anat/sub-01_dseg.nii.gz
            reference: ${register.args.output}
            output: /out/sub-01/seg_resampled.nii.gz

        # Level 3 — waits for register AND resample_seg
        - name: write_tacs
          command: petpal-write-tacs
          depends_on: [register, resample_seg]
          args:
            pet: ${register.args.output}
            seg: ${resample_seg.args.output}
            output: /out/sub-01/tacs.tsv

        # Level 4 — patlak and logan run IN PARALLEL (same level,
        #           independent of each other, both wait for write_tacs)
        - name: patlak
          command: petpal-graph-analysis
          depends_on: [write_tacs]
          args:
            method: patlak
            tac: ${write_tacs.args.output}
            output: /out/sub-01/patlak_ki.nii.gz

        - name: logan
          command: petpal-graph-analysis
          depends_on: [write_tacs]
          args:
            method: logan
            tac: ${write_tacs.args.output}
            output: /out/sub-01/logan_dvr.nii.gz

Run it::

    petpal-run examples/full_pet_pipeline.yaml --workers 4 --log-level INFO

Expected log output (abbreviated)::

    [INFO]  Level 1/5 — submitting: ['moco']
    [INFO]  Completed 'moco'.
    [INFO]  Level 1/5 — all steps completed.
    [INFO]  Level 2/5 — submitting: ['register']
    [INFO]  Completed 'register'.
    ...
    [INFO]  Level 5/5 — submitting: ['patlak', 'logan']
    [INFO]  Completed 'patlak'.
    [INFO]  Completed 'logan'.
    [INFO]  Level 5/5 — all steps completed.


Bash Scripting Without a Pipeline File
---------------------------------------

Because every scientific function has its own CLI, you can also compose
pipelines as plain bash scripts without the YAML runner:

.. code-block:: bash

    #!/usr/bin/env bash
    set -euo pipefail

    PET=/data/sub-01/pet/sub-01_trc-FDG_pet.nii.gz
    T1=/data/sub-01/anat/sub-01_T1w.nii.gz
    SEG=/data/sub-01/anat/sub-01_dseg.nii.gz
    OUT=/out/sub-01

    petpal-moco windowed \
        --pet "$PET" \
        --output "$OUT/moco.nii.gz"

    petpal-register to-t1 \
        --pet "$OUT/moco.nii.gz" \
        --reference "$T1" \
        --output "$OUT/reg.nii.gz"

    petpal-seg resample \
        --seg "$SEG" \
        --reference "$OUT/reg.nii.gz" \
        --output "$OUT/seg_resampled.nii.gz"

    petpal-write-tacs \
        --pet "$OUT/reg.nii.gz" \
        --seg "$OUT/seg_resampled.nii.gz" \
        --output "$OUT/tacs.tsv"

    # Run Patlak and Logan in parallel with bash '&' and 'wait'
    petpal-graph-analysis --method patlak \
        --tac "$OUT/tacs.tsv" --output "$OUT/patlak_ki.nii.gz" &

    petpal-graph-analysis --method logan \
        --tac "$OUT/tacs.tsv" --output "$OUT/logan_dvr.nii.gz" &

    wait   # wait for both background jobs
    echo "Pipeline complete."

The YAML runner is recommended for complex pipelines with many steps; bash
scripts are ideal for quick one-off analyses or when maximum control is needed.
