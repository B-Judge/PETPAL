"""pytest configuration for PETPAL tests.

Stubs out heavy scientific and medical-imaging dependencies (antspyx,
nibabel, SimpleITK, numba, fsl, sklearn, etc.) into ``sys.modules``
*before* any test file imports ``petpal``.  The top-level
``petpal/__init__.py`` eagerly imports every subpackage, each of which
pulls in one or more of these packages; inserting lightweight
:class:`unittest.mock.MagicMock` objects prevents ImportError in
environments where those packages are not installed (e.g. plain CI
runners or the pipeline-runner unit-test environment).

Tests that exercise code genuinely requiring one of these libraries
should be marked ``@pytest.mark.requires_ants``,
``@pytest.mark.requires_nibabel``, etc., and skipped when the real
package is absent.

Design note — why no ``spec``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``MagicMock(spec=types.ModuleType(name))`` restricts attribute access to the
attributes actually present on a blank ``ModuleType`` instance.  This causes
``from bids_validator import BIDSValidator`` (and similar ``from x import y``
statements) to raise ``ImportError`` because ``BIDSValidator`` is not a known
attribute of a blank module.  Using ``MagicMock()`` without a spec allows any
attribute lookup, which is the correct behaviour for a module stub.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stub(name: str, attrs: dict | None = None) -> MagicMock:
    """Create a spec-free MagicMock that impersonates a module.

    Using no ``spec`` (rather than ``spec=types.ModuleType(name)``) is
    intentional: it allows ``from <stub> import <anything>`` to succeed,
    which is required for modules like ``bids_validator`` that are imported
    with explicit ``from`` imports in petpal source files.

    Args:
        name: Fully-qualified module name (e.g. ``"ants"``).
        attrs: Optional mapping of attribute names to preset values.

    Returns:
        A spec-free MagicMock configured as a module stub.
    """
    stub = MagicMock()
    stub.__name__ = name
    stub.__spec__ = None
    stub.__package__ = name.split(".")[0]
    stub.__path__ = []   # marks it as a package so sub-imports resolve
    stub.__file__ = None
    if attrs:
        for attr, val in attrs.items():
            setattr(stub, attr, val)
    return stub


def _install(name: str, attrs: dict | None = None) -> MagicMock:
    """Insert a stub into ``sys.modules`` if the real package is absent.

    Args:
        name: Module name to stub.
        attrs: Optional attribute overrides forwarded to :func:`_make_stub`.

    Returns:
        The stub (existing entry if already present).
    """
    if name not in sys.modules:
        sys.modules[name] = _make_stub(name, attrs)
    return sys.modules[name]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# pytest hook — runs before test collection
# ---------------------------------------------------------------------------


def pytest_configure(config) -> None:  # noqa: ANN001
    """Install lightweight stubs for every heavy scientific dependency.

    Called by pytest before any test module is imported, ensuring that
    ``import petpal.*`` succeeds without the real packages being present.
    """
    _STUBS: list[str] = [
        # ANTs / antspyx
        "ants",
        "ants.core",
        "ants.core.ants_image",
        "ants.registration",
        "ants.utils",
        "ants.utils.convert_nibabel",
        "antspyx",
        # nibabel
        "nibabel",
        "nibabel.nifti1",
        "nibabel.nifti2",
        "nibabel.processing",
        "nibabel.orientations",
        "nibabel.affines",
        "nibabel.loadsave",
        # SimpleITK
        "SimpleITK",
        # numba
        "numba",
        "numba.core",
        "numba.typed",
        # FSL
        "fsl",
        "fsl.wrappers",
        "fsl.data",
        "fsl.data.image",
        "fsl.utils",
        "fsl.utils.run",
        "fslpy",
        # scikit-image
        "skimage",
        "skimage.filters",
        "skimage.morphology",
        "skimage.measure",
        "skimage.transform",
        # scikit-learn
        "sklearn",
        "sklearn.decomposition",
        "sklearn.preprocessing",
        "sklearn.pipeline",
        "sklearn.utils",
        # lmfit
        "lmfit",
        "lmfit.models",
        # docker
        "docker",
        "docker.errors",
        # bids_validator
        "bids_validator",
        # nilearn
        "nilearn",
        "nilearn.image",
        # dipy
        "dipy",
        "dipy.align",
    ]

    stubs: dict[str, MagicMock] = {}
    for mod_name in _STUBS:
        stubs[mod_name] = _install(mod_name)

    # Wire child stubs onto their parent stubs so that attribute access like
    # ``fsl.wrappers.applywarp`` resolves without AttributeError.
    _PARENT_CHILD: list[tuple[str, str]] = [
        ("ants", "core"),
        ("ants", "registration"),
        ("ants", "utils"),
        ("ants.core", "ants_image"),
        ("ants.utils", "convert_nibabel"),
        ("nibabel", "nifti1"),
        ("nibabel", "nifti2"),
        ("nibabel", "processing"),
        ("nibabel", "orientations"),
        ("nibabel", "affines"),
        ("nibabel", "loadsave"),
        ("numba", "core"),
        ("numba", "typed"),
        ("fsl", "wrappers"),
        ("fsl", "data"),
        ("fsl", "utils"),
        ("fsl.data", "image"),
        ("fsl.utils", "run"),
        ("skimage", "filters"),
        ("skimage", "morphology"),
        ("skimage", "measure"),
        ("skimage", "transform"),
        ("sklearn", "decomposition"),
        ("sklearn", "preprocessing"),
        ("sklearn", "pipeline"),
        ("sklearn", "utils"),
        ("lmfit", "models"),
        ("docker", "errors"),
        ("nilearn", "image"),
        ("dipy", "align"),
    ]

    for parent_name, child_attr in _PARENT_CHILD:
        child_full = f"{parent_name}.{child_attr}"
        parent_stub = stubs.get(parent_name)
        child_stub = stubs.get(child_full)
        if parent_stub is not None and child_stub is not None:
            setattr(parent_stub, child_attr, child_stub)
