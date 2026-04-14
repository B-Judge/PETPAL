"""Standalone CLI for PCA-guided image-derived input function (IDIF) extraction.

Entry point: ``petpal-pca-idif``

Subcommands
-----------
top-voxels
    Extract an IDIF from the top-scoring voxels of a chosen PCA component.
    Simple and fast; good starting point.
fitter
    Extract an IDIF via PCA-guided quantile optimisation using lmfit.
    More robust but slower.

Examples
--------
::

    petpal-pca-idif top-voxels \\
        --pet sub-01_pet.nii.gz \\
        --mask carotid_mask.nii.gz \\
        --output idif.tsv

    petpal-pca-idif fitter \\
        --pet sub-01_pet.nii.gz \\
        --mask carotid_mask.nii.gz \\
        --output idif.tsv \\
        --alpha 1.0 --beta 1.0
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..input_function.pca_guided_idif import PCAGuidedIdifFitter, PCAGuidedTopVoxelsIDIF


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="petpal-pca-idif",
        description="Extract an image-derived input function (IDIF) using PCA guidance.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- top-voxels
    tv = sub.add_parser(
        "top-voxels",
        help="IDIF from the top-scoring voxels of a PCA component.",
    )
    tv.add_argument("--pet", required=True, type=Path, metavar="PATH",
                    help="Input 4-D PET image.")
    tv.add_argument("--mask", required=True, type=Path, metavar="PATH",
                    help="Binary mask defining the IDIF search region (e.g. carotids).")
    tv.add_argument("--output", required=True, type=Path, metavar="PATH",
                    help="Output IDIF TAC file (.tsv).")
    tv.add_argument("--n-components", type=int, default=5, metavar="N",
                    help="Number of PCA components to compute (default: 5).")
    tv.add_argument("--component", type=int, default=0, metavar="IDX",
                    help="PCA component index used for voxel selection (default: 0).")
    tv.add_argument("--n-voxels", type=int, default=100, metavar="N",
                    help="Number of top-scoring voxels to include (default: 100).")
    tv.add_argument("--verbose", action="store_true")

    # -- fitter
    fit = sub.add_parser(
        "fitter",
        help="IDIF via PCA-guided quantile optimisation.",
    )
    fit.add_argument("--pet", required=True, type=Path, metavar="PATH",
                     help="Input 4-D PET image.")
    fit.add_argument("--mask", required=True, type=Path, metavar="PATH",
                     help="Binary mask defining the IDIF search region.")
    fit.add_argument("--output", required=True, type=Path, metavar="PATH",
                     help="Output IDIF TAC file (.tsv).")
    fit.add_argument("--n-components", type=int, default=5, metavar="N",
                     help="Number of PCA components (default: 5).")
    fit.add_argument("--alpha", type=float, default=1.0, metavar="FLOAT",
                     help="Noise/smoothness trade-off weight (default: 1.0).")
    fit.add_argument("--beta", type=float, default=1.0, metavar="FLOAT",
                     help="Peak term weight (default: 1.0).")
    fit.add_argument("--method", default="ampgo", metavar="METHOD",
                     help="lmfit optimisation method (default: ampgo).")
    fit.add_argument("--min-filter-value", type=float, default=0.0, metavar="FLOAT",
                     help="PCA component filter minimum value (default: 0.0).")
    fit.add_argument("--filter-threshold", type=float, default=0.1, metavar="FLOAT",
                     help="PCA component filter threshold (default: 0.1).")
    fit.add_argument("--verbose", action="store_true")

    return parser


def main() -> None:
    """Entry point for ``petpal-pca-idif``."""
    args = _build_parser().parse_args()

    if args.command == "top-voxels":
        idif = PCAGuidedTopVoxelsIDIF(
            input_image_path=str(args.pet),
            mask_image_path=str(args.mask),
            output_tac_path=str(args.output),
            num_pca_components=args.n_components,
            verbose=args.verbose,
        )
        idif.run(selected_component=args.component, num_of_voxels=args.n_voxels)
    elif args.command == "fitter":
        idif = PCAGuidedIdifFitter(
            input_image_path=str(args.pet),
            mask_image_path=str(args.mask),
            output_tac_path=str(args.output),
            num_pca_components=args.n_components,
            pca_comp_filter_min_value=args.min_filter_value,
            pca_comp_threshold=args.filter_threshold,
            verbose=args.verbose,
        )
        idif.run(alpha=args.alpha, beta=args.beta, method=args.method)


if __name__ == "__main__":
    main()
