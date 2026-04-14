"""Standalone CLI for SUV and SUVR calculations.

Entry point: ``petpal-suv``

Subcommands
-----------
suv
    Compute a standardized uptake value image from a 4-D PET scan.
suvr
    Compute an SUV ratio image using a segmentation-defined reference region.
weighted-sum
    Compute a frame-duration-weighted mean image (useful as input to SUVR).

Examples
--------
::

    petpal-suv suv \\
        --pet sub-01_pet.nii.gz \\
        --output sub-01_suv.nii.gz \\
        --weight 75.0 --dose 370.0 \\
        --start-time 40.0 --end-time 60.0

    petpal-suv suvr \\
        --pet sub-01_pet.nii.gz \\
        --output sub-01_suvr.nii.gz \\
        --seg sub-01_dseg.nii.gz \\
        --ref-region 41 42 \\
        --start-time 40.0 --end-time 60.0
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..preproc.standard_uptake_value import suv, suvr, weighted_sum_for_suv


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="petpal-suv",
        description="Compute SUV, SUVR, or weighted-sum images from a 4-D PET scan.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- suv
    suv_p = sub.add_parser("suv", help="Compute standardized uptake value image.")
    suv_p.add_argument("--pet", required=True, type=Path, metavar="PATH",
                       help="Input 4-D PET image.")
    suv_p.add_argument("--output", required=True, type=Path, metavar="PATH",
                       help="Output SUV image.")
    suv_p.add_argument("--weight", required=True, type=float, metavar="KG",
                       help="Subject weight in kilograms.")
    suv_p.add_argument("--dose", required=True, type=float, metavar="MBQ",
                       help="Injected dose in MBq.")
    suv_p.add_argument("--start-time", required=True, type=float, metavar="MIN",
                       help="Integration start time in minutes.")
    suv_p.add_argument("--end-time", required=True, type=float, metavar="MIN",
                       help="Integration end time in minutes.")

    # -- suvr
    suvr_p = sub.add_parser("suvr", help="Compute SUV ratio image.")
    suvr_p.add_argument("--pet", required=True, type=Path, metavar="PATH",
                        help="Input 4-D PET image.")
    suvr_p.add_argument("--output", required=True, type=Path, metavar="PATH",
                        help="Output SUVR image.")
    suvr_p.add_argument("--seg", required=True, type=Path, metavar="PATH",
                        help="Segmentation image defining the reference region.")
    suvr_p.add_argument("--ref-region", required=True, type=int, nargs="+",
                        metavar="LABEL",
                        help="Segmentation label(s) for the reference region.")
    suvr_p.add_argument("--start-time", required=True, type=float, metavar="MIN",
                        help="Integration start time in minutes.")
    suvr_p.add_argument("--end-time", required=True, type=float, metavar="MIN",
                        help="Integration end time in minutes.")

    # -- weighted-sum
    ws_p = sub.add_parser(
        "weighted-sum",
        help="Compute a frame-duration-weighted mean image.",
    )
    ws_p.add_argument("--pet", required=True, type=Path, metavar="PATH",
                      help="Input 4-D PET image.")
    ws_p.add_argument("--output", type=Path, metavar="PATH", default=None,
                      help="Output image path (optional).")
    ws_p.add_argument("--start-time", type=float, default=0.0, metavar="MIN",
                      help="Integration start time in minutes (default: 0).")
    ws_p.add_argument("--end-time", type=float, default=-1.0, metavar="MIN",
                      help="Integration end time in minutes; -1 = end of scan (default: -1).")

    return parser


def main() -> None:
    """Entry point for ``petpal-suv``."""
    args = _build_parser().parse_args()

    if args.command == "suv":
        suv(
            input_image_path=str(args.pet),
            output_image_path=str(args.output),
            weight=args.weight,
            dose=args.dose,
            start_time=args.start_time,
            end_time=args.end_time,
        )
    elif args.command == "suvr":
        ref = args.ref_region[0] if len(args.ref_region) == 1 else args.ref_region
        suvr(
            input_image_path=str(args.pet),
            output_image_path=str(args.output),
            segmentation_image_path=str(args.seg),
            ref_region=ref,
            start_time=args.start_time,
            end_time=args.end_time,
        )
    elif args.command == "weighted-sum":
        weighted_sum_for_suv(
            input_image_path=str(args.pet),
            output_image_path=str(args.output) if args.output else None,
            start_time=args.start_time,
            end_time=args.end_time,
        )


if __name__ == "__main__":
    main()
