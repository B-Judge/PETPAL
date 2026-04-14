"""Standalone CLI for PET radioactive decay correction.

Entry point: ``petpal-decay-correct``

Subcommands
-----------
apply
    Apply decay correction to a PET image.
undo
    Remove decay correction from a PET image (e.g. before kinetic modelling
    that expects non-decay-corrected data).

Examples
--------
::

    petpal-decay-correct apply \\
        --input sub-01_pet.nii.gz \\
        --output sub-01_pet_dc.nii.gz

    petpal-decay-correct undo \\
        --input sub-01_pet_dc.nii.gz \\
        --output sub-01_pet_nodecay.nii.gz
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..preproc.decay_correction import decay_correct, undo_decay_correction


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="petpal-decay-correct",
        description="Apply or undo PET radioactive decay correction.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    apply_p = sub.add_parser("apply", help="Apply decay correction.")
    apply_p.add_argument("--input", required=True, type=Path, metavar="PATH",
                         help="Input PET image.")
    apply_p.add_argument("--output", required=True, type=Path, metavar="PATH",
                         help="Output decay-corrected image.")

    undo_p = sub.add_parser("undo", help="Remove decay correction.")
    undo_p.add_argument("--input", required=True, type=Path, metavar="PATH",
                        help="Input decay-corrected PET image.")
    undo_p.add_argument("--output", required=True, type=Path, metavar="PATH",
                        help="Output image with decay correction removed.")

    return parser


def main() -> None:
    """Entry point for ``petpal-decay-correct``."""
    args = _build_parser().parse_args()

    if args.command == "apply":
        decay_correct(
            input_image_path=str(args.input),
            output_image_path=str(args.output),
        )
    elif args.command == "undo":
        undo_decay_correction(
            input_image_path=str(args.input),
            output_image_path=str(args.output),
        )


if __name__ == "__main__":
    main()
