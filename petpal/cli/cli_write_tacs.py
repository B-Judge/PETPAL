"""Standalone CLI for extracting regional TACs from a PET image.

Entry point: ``petpal-write-tacs``

Extracts mean time-activity curves (TACs) for every region defined in a
segmentation image, using a label map to assign human-readable names.  One
``.tsv`` file is written per region into the output directory.

Examples
--------
::

    petpal-write-tacs \\
        --pet sub-01_pet.nii.gz \\
        --seg sub-01_dseg.nii.gz \\
        --label-map freesurfer_lmap.json \\
        --output-dir tacs/ \\
        --prefix sub-01_ses-baseline_
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..preproc.regional_tac_extraction import write_tacs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="petpal-write-tacs",
        description=(
            "Extract regional TACs from a 4-D PET image using a segmentation "
            "and label map."
        ),
    )
    parser.add_argument("--pet", required=True, type=Path, metavar="PATH",
                        help="Input 4-D PET image.")
    parser.add_argument("--seg", required=True, type=Path, metavar="PATH",
                        help="Segmentation image (integer label map).")
    parser.add_argument("--label-map", required=True, type=Path, metavar="PATH",
                        help="JSON or TSV file mapping integer labels to region names.")
    parser.add_argument("--output-dir", required=True, type=Path, metavar="DIR",
                        help="Output directory for TAC .tsv files.")
    parser.add_argument("--prefix", default="", metavar="PREFIX",
                        help="Filename prefix for all output TAC files (default: none).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose output.")
    return parser


def main() -> None:
    """Entry point for ``petpal-write-tacs``."""
    args = _build_parser().parse_args()
    write_tacs(
        input_image_path=str(args.pet),
        label_map_path=str(args.label_map),
        segmentation_image_path=str(args.seg),
        out_tac_dir=str(args.output_dir),
        out_tac_prefix=args.prefix,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
