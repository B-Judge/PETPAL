"""Standalone CLI for PET image registration.

Entry point: ``petpal-register``

Subcommands
-----------
to-t1
    Register a PET image to a T1-weighted anatomical reference using ANTs.
    This is the standard step after motion correction.
to-atlas
    Warp a PET image into a standard atlas space via the T1 anatomical image.
apply-xfm
    Apply a pre-computed ANTs transform to an image.

Examples
--------
::

    petpal-register to-t1 \\
        --pet moco.nii.gz \\
        --reference sub-01_T1w.nii.gz \\
        --output reg.nii.gz

    petpal-register to-atlas \\
        --pet reg.nii.gz \\
        --anat sub-01_T1w.nii.gz \\
        --atlas MNI152.nii.gz
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..preproc.register import apply_xfm_ants, register_pet, warp_pet_to_atlas


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="petpal-register",
        description="PET image registration.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- to-t1
    t1 = sub.add_parser(
        "to-t1",
        help="Register PET to a T1 anatomical reference.",
    )
    t1.add_argument("--pet", required=True, type=Path, metavar="PATH",
                    help="Input PET image to register.")
    t1.add_argument("--reference", required=True, type=Path, metavar="PATH",
                    help="Reference anatomical image (e.g. T1w).")
    t1.add_argument("--output", required=True, type=Path, metavar="PATH",
                    help="Output registered PET image.")
    t1.add_argument("--motion-target", default="mean", metavar="TARGET",
                    help="Motion target used prior to registration (default: mean).")
    t1.add_argument("--transform", default="DenseRigid", metavar="TYPE",
                    help="ANTs transform type (default: DenseRigid).")
    t1.add_argument("--verbose", action="store_true",
                    help="Print verbose registration output.")

    # -- to-atlas
    atlas = sub.add_parser(
        "to-atlas",
        help="Warp PET to a standard atlas space.",
    )
    atlas.add_argument("--pet", required=True, type=Path, metavar="PATH",
                       help="Input PET image.")
    atlas.add_argument("--anat", required=True, type=Path, metavar="PATH",
                       help="T1 anatomical image in the same space as the PET.")
    atlas.add_argument("--atlas", required=True, type=Path, metavar="PATH",
                       help="Target atlas image.")
    atlas.add_argument("--transform", default="SyN", metavar="TYPE",
                       help="ANTs transform type (default: SyN).")

    # -- apply-xfm
    xfm = sub.add_parser(
        "apply-xfm",
        help="Apply a pre-computed ANTs transform to an image.",
    )
    xfm.add_argument("--input", required=True, type=Path, metavar="PATH",
                     help="Input image.")
    xfm.add_argument("--reference", required=True, type=Path, metavar="PATH",
                     help="Reference image defining the output space.")
    xfm.add_argument("--output", required=True, type=Path, metavar="PATH",
                     help="Output warped image.")
    xfm.add_argument("--transforms", required=True, nargs="+", metavar="PATH",
                     help="One or more ANTs transform files applied in order.")
    xfm.add_argument("--copy-meta", action="store_true",
                     help="Copy metadata from input to output.")

    return parser


def main() -> None:
    """Entry point for ``petpal-register``."""
    args = _build_parser().parse_args()

    if args.command == "to-t1":
        register_pet(
            input_reg_image_path=str(args.pet),
            out_image_path=str(args.output),
            reference_image_path=str(args.reference),
            motion_target_option=args.motion_target,
            verbose=args.verbose,
            type_of_transform=args.transform,
        )
    elif args.command == "to-atlas":
        warp_pet_to_atlas(
            input_image_path=str(args.pet),
            anat_image_path=str(args.anat),
            atlas_image_path=str(args.atlas),
            type_of_transform=args.transform,
        )
    elif args.command == "apply-xfm":
        apply_xfm_ants(
            input_image_path=str(args.input),
            ref_image_path=str(args.reference),
            out_image_path=str(args.output),
            xfm_paths=[str(p) for p in args.transforms],
            copy_meta=args.copy_meta,
        )


if __name__ == "__main__":
    main()
