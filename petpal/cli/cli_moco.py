"""Standalone CLI for PET motion correction.

Entry point: ``petpal-moco``

Subcommands
-----------
frames-above-mean
    Motion-correct only frames whose mean intensity exceeds the whole-scan
    mean.  This is the standard preprocessing approach for most PET scans.
windowed
    Motion-correct using a rolling temporal window.  Useful for long dynamic
    scans with gradual head movement.

Examples
--------
::

    petpal-moco frames-above-mean \\
        --pet sub-01_pet.nii.gz \\
        --output moco.nii.gz

    petpal-moco windowed \\
        --pet sub-01_pet.nii.gz \\
        --output moco.nii.gz \\
        --window-size 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..preproc.motion_corr import (
    motion_corr_frames_above_mean_value,
    windowed_motion_corr_to_target,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="petpal-moco",
        description="PET image motion correction.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- frames-above-mean
    fam = sub.add_parser(
        "frames-above-mean",
        help="Motion-correct frames above the scan-mean intensity.",
    )
    fam.add_argument("--pet", required=True, type=Path, metavar="PATH",
                     help="Input 4-D PET image.")
    fam.add_argument("--output", required=True, type=Path, metavar="PATH",
                     help="Output motion-corrected image.")
    fam.add_argument("--target", default="mean", metavar="TARGET",
                     help="Motion target: 'mean', 'median', or a frame index (default: mean).")
    fam.add_argument("--transform", default="Affine",
                     choices=["Rigid", "Affine", "DenseRigid"],
                     help="ANTs transform type (default: Affine).")
    fam.add_argument("--metric", default="mattes", metavar="METRIC",
                     help="ANTs registration metric (default: mattes).")
    fam.add_argument("--scale-factor", type=float, default=1.0, metavar="FLOAT",
                     help="Intensity scale factor applied to the target frame (default: 1.0).")
    fam.add_argument("--verbose", action="store_true",
                     help="Print verbose registration output.")

    # -- windowed
    win = sub.add_parser(
        "windowed",
        help="Motion-correct using a rolling temporal window.",
    )
    win.add_argument("--pet", required=True, type=Path, metavar="PATH",
                     help="Input 4-D PET image.")
    win.add_argument("--output", required=True, type=Path, metavar="PATH",
                     help="Output motion-corrected image.")
    win.add_argument("--target", default="mean", metavar="TARGET",
                     help="Motion target: 'mean', 'median', or a frame index (default: mean).")
    win.add_argument("--window-size", required=True, type=float, metavar="MINUTES",
                     help="Window size in minutes.")
    win.add_argument("--transform", default="QuickRigid", metavar="TYPE",
                     help="ANTs transform type (default: QuickRigid).")
    win.add_argument("--verbose", action="store_true",
                     help="Print verbose registration output.")

    return parser


def main() -> None:
    """Entry point for ``petpal-moco``."""
    args = _build_parser().parse_args()

    if args.command == "frames-above-mean":
        motion_corr_frames_above_mean_value(
            input_image_path=str(args.pet),
            out_image_path=str(args.output),
            motion_target_option=args.target,
            verbose=args.verbose,
            type_of_transform=args.transform,
            transform_metric=args.metric,
            scale_factor=args.scale_factor,
        )
    elif args.command == "windowed":
        windowed_motion_corr_to_target(
            input_image_path=str(args.pet),
            out_image_path=str(args.output),
            motion_target_option=args.target,
            w_size=args.window_size,
            type_of_transform=args.transform,
        )


if __name__ == "__main__":
    main()
