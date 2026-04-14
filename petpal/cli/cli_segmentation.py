"""Standalone CLI for brain segmentation utilities.

Entry point: ``petpal-seg``

Subcommands
-----------
resample
    Resample a segmentation image to match the voxel grid of a PET image.
erode-wm
    Create an eroded white-matter reference region segmentation.
vat-wm
    Generate a VAT white-matter reference region segmentation.

Examples
--------
::

    petpal-seg resample \\
        --pet moco.nii.gz \\
        --seg sub-01_dseg.nii.gz \\
        --output sub-01_dseg_res.nii.gz

    petpal-seg erode-wm \\
        --seg sub-01_dseg.nii.gz \\
        --output sub-01_wm_eroded.nii.gz
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..preproc.segmentation_tools import (
    eroded_wm_segmentation,
    resample_segmentation,
    vat_wm_ref_region,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="petpal-seg",
        description="Brain segmentation utilities for PET preprocessing.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- resample
    res = sub.add_parser(
        "resample",
        help="Resample a segmentation image to match a PET image.",
    )
    res.add_argument("--pet", required=True, type=Path, metavar="PATH",
                     help="Reference PET image (defines the target voxel grid).")
    res.add_argument("--seg", required=True, type=Path, metavar="PATH",
                     help="Input segmentation image.")
    res.add_argument("--output", required=True, type=Path, metavar="PATH",
                     help="Output resampled segmentation.")
    res.add_argument("--verbose", action="store_true")

    # -- erode-wm
    ewm = sub.add_parser(
        "erode-wm",
        help="Create an eroded white-matter reference region segmentation.",
    )
    ewm.add_argument("--seg", required=True, type=Path, metavar="PATH",
                     help="Input segmentation image.")
    ewm.add_argument("--output", required=True, type=Path, metavar="PATH",
                     help="Output eroded WM segmentation.")
    ewm.add_argument("--wm-label", type=int, default=1, metavar="LABEL",
                     help="Integer label assigned to the eroded WM region (default: 1).")

    # -- vat-wm
    vat = sub.add_parser(
        "vat-wm",
        help="Generate a VAT white-matter reference region segmentation.",
    )
    vat.add_argument("--seg", required=True, type=Path, metavar="PATH",
                     help="Input segmentation image.")
    vat.add_argument("--output", required=True, type=Path, metavar="PATH",
                     help="Output VAT WM reference region image.")

    return parser


def main() -> None:
    """Entry point for ``petpal-seg``."""
    args = _build_parser().parse_args()

    if args.command == "resample":
        resample_segmentation(
            input_image_path=str(args.pet),
            segmentation_image_path=str(args.seg),
            out_seg_path=str(args.output),
            verbose=args.verbose,
        )
    elif args.command == "erode-wm":
        eroded_wm_segmentation(
            input_segmentation_path=str(args.seg),
            out_segmentation_path=str(args.output),
            eroded_wm_region_mapping=args.wm_label,
        )
    elif args.command == "vat-wm":
        vat_wm_ref_region(
            input_segmentation_path=str(args.seg),
            out_segmentation_path=str(args.output),
        )


if __name__ == "__main__":
    main()
