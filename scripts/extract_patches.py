#!/usr/bin/env python
"""Script to extracting patches from total-ion-current images."""
from argparse import Namespace, ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
from time import perf_counter

from pandas import DataFrame

from ispy_msi.patch_extraction_utils import (  # type: ignore
    tessellate_image_mask_pair,
)


def _parse_command_line() -> Namespace:
    """Parse the command-line arguments.

    Returns
    -------
    Namespace
        Command-line arguments in a ``Namespace``.

    """
    parser = ArgumentParser(
        description="Extract patches from total-ion-current images.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "image_dir",
        help="Directory containing 'images/' and 'masks/'.",
        type=Path,
    )

    parser.add_argument(
        "--patch-size",
        help="Linear size (in pixels) of square patches.",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--stride",
        help="Stride to use in the patch extraction.",
        type=int,
        default=64,
    )

    return parser.parse_args()


def _get_image_and_mask_data(parent_dir: Path) -> DataFrame:
    """List the images and masks.

    Parameters
    ----------
    parent_dir : Path
        Base directory the data are stored in.

    Returns
    -------
    data : DataFrame
        Corresponding image and mask paths

    """
    data = DataFrame()
    data["mask"] = sorted((parent_dir / "masks").glob("*.png"), key=lambda x: x.name)
    data["image"] = sorted((parent_dir / "images").glob("*.png"), key=lambda x: x.name)

    return data


def _extract_patches(args: Namespace):
    """Extract patches from total-ion-current-images.

    Parameters
    ----------
    args : Namespace
        The command-line arguments.

    """
    data = _get_image_and_mask_data(args.image_dir)

    for row in data.itertuples():

        start = perf_counter()

        tessellate_image_mask_pair(
            args.image_dir,
            row.image,  # type: ignore
            row.mask,  # type: ignore
            args.patch_size,
            args.stride,
        )

        stop = perf_counter()

        print(f"Processed '{row.image.stem}' in {stop - start:.6f} seconds.")  # type: ignore


if __name__ == "__main__":
    _extract_patches(_parse_command_line())
