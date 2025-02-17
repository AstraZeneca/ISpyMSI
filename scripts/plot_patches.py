#!/usr/bin/env python
"""Script for saving some random patches to image file."""

from pathlib import Path

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace

from torch import from_numpy  # pylint: disable=no-name-in-module
from torchvision.transforms import RandomCrop  # type: ignore

from numpy import load, concatenate

from skimage.io import imsave
from skimage.util import img_as_ubyte

from ispy_msi.patch_extraction_utils import (
    preprocess_tic_img,
    preprocess_rms_img,
    preprocess_entropy_img,
)


def _parse_command_line() -> Namespace:
    """Parse the command-line arguments.

    Returns
    -------
    Namespace
        Command-line arguments.

    """
    parser = ArgumentParser(
        description="Create some patch visuals for a figure.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "ion_dir",
        help="Directory containing the ion images.",
        type=Path,
    )

    parser.add_argument(
        "--patch-size",
        help="Length of the square patches to save.",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--save-dir",
        help="Directory to save the images in.",
        type=Path,
        default="patch-visuals",
    )

    return parser.parse_args()


def plot_random_patches(args: Namespace):
    """Plot random patches for a figure visualisation.

    Parameters
    ----------
    args : Namespace
        Command-line arguments.

    """
    cropper = RandomCrop(args.patch_size)

    args.save_dir.mkdir(exist_ok=True, parents=True)

    for img_path in list(args.ion_dir.glob("*/tic.npy")):

        chans = [
            preprocess_tic_img(load(img_path))[:, :, None],
            preprocess_rms_img(load(img_path.with_name("rms.npy")))[:, :, None],
            preprocess_entropy_img(load(img_path.with_name("entropy.npy")))[:, :, None],
        ]

        img = concatenate(chans, axis=2)

        crop = cropper(from_numpy(img).permute(2, 0, 1)).permute(1, 2, 0).numpy()

        rgb = img_as_ubyte(crop)

        save_path = args.save_dir / Path(img_path.parent.name).with_suffix(".png")

        imsave(save_path, rgb)


if __name__ == "__main__":
    plot_random_patches(_parse_command_line())
