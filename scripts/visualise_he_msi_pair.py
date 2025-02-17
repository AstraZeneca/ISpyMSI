#!/usr/bin/env python
"""Visualise H&E-MSI pairs."""
from typing import List

from pathlib import Path

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace

from pandas import read_csv

from numpy import ndarray, array, pad, concatenate, load

from tiffslide import TiffSlide

from skimage.transform import rescale, resize  # pylint: disable=no-name-in-module

import matplotlib.pyplot as plt


def _parse_command_line() -> Namespace:
    """Parse the command-line arguments.

    Returns
    -------
    Namespace
        The command-line arguments.

    """
    parser = ArgumentParser(
        description="Visualise H&E-MSI pairs.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "base_dir",
        help="Project base directory with all data.",
        type=Path,
    )

    parser.add_argument(
        "msi_name",
        help="Name of the MSI to visualise.",
        type=str,
    )

    parser.add_argument(
        "--metab",
        help="Name of the metabolite to show",
        type=str,
        default="Lactate",
    )

    parser.add_argument(
        "--level",
        help="Level in the tiff to extract at.",
        type=int,
        default=2,
    )

    return parser.parse_args()


def _load_single_he(he_path: Path, level: int, target_mpp: float) -> ndarray:
    """Load a single low-power overview of an H&E.

    Parameters
    ----------
    he_path : Path
        Path to the whole-slide image.
    level : int
        Level in the tiff to take the overview image at.
    target_mpp : float
        The target MPP to return the overview image at.

    """
    with TiffSlide(he_path) as slide:

        zero_mpp = float(slide.properties["aperio.MPP"])
        downsample = slide.level_downsamples[level]

        extracted_mpp = zero_mpp * downsample

        overview = slide.read_region(
            location=(0, 0),
            level=level,
            size=slide.level_dimensions[level],
            as_array=True,
        )

    scale_factor = target_mpp / extracted_mpp

    return rescale(overview, scale_factor**-1.0, channel_axis=2)


def combine_hes(
    he_paths: List[Path],
    target_mpp: float,
    level: int,
) -> ndarray:
    """Concatenate low-power overview of the H&Es at ``target_mpp``.

    Parameters
    ----------
    he_paths : List[Path]
        Paths to the H&E images.
    target_mpp : float
        The target microns per pixel.
    level : int
        Level in the tiff to extract the overview at.

    """
    he_imgs: List[ndarray] = []
    for he_path in he_paths:
        overview = _load_single_he(he_path, level, target_mpp)
        he_imgs.append(overview)

    dims = array(list(map(lambda x: x.shape, he_imgs)))
    pads = abs(dims - dims.max(axis=0))

    # We only care about width
    pads[:, [0, 2]] = 0

    padded = []
    for he_img, pad_amount in zip(he_imgs, pads):

        pad_with = list(zip((0, 0, 0), list(pad_amount)))

        padded.append(pad(he_img, pad_width=pad_with, constant_values=1.0))

    return concatenate(padded, axis=0)


def _load_msi(msi_path: Path) -> ndarray:
    """Load an MSI channel to visulise.

    Parameters
    ----------
    msi_path : Path
        Path to the MSI.

    Returns
    -------
    ndarray
        RMS-normalised ion image.

    """
    return load(msi_path) / load(msi_path.with_name("rms.npy"))


def produce_visuals(args: Namespace):
    """Produce the H&E-MSI visuals.

    Parameters
    ----------
    args : Namespace
        Command-line arguments.

    """
    metadata = read_csv(args.base_dir / "metadata.csv")
    metadata = metadata.loc[metadata.msi == args.msi_name]

    he_paths = list(
        map(lambda x: args.base_dir / f"H&E/{x}", metadata.he_img.to_list())
    )[::-1]

    he_img = combine_hes(
        he_paths,
        metadata.msi_microns_pp.unique().item(),
        args.level,
    )

    msi = _load_msi(
        args.base_dir / f"ion-imgs/{args.msi_name}/{args.metab}.npy",
    )

    he_img = resize(he_img, msi.shape[:2])

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(3.0, 3.0 * msi.shape[1] / msi.shape[0]),
    )
    axes[0].imshow(he_img)  # type: ignore
    axes[1].imshow(msi, cmap="inferno")  # type: ignore

    for axis in axes.ravel():  # type: ignore
        axis.set_xticks([])
        axis.set_yticks([])

    figure.tight_layout(pad=0.0)

    figure.savefig("pair-vis.pdf", dpi=250)


if __name__ == "__main__":
    produce_visuals(_parse_command_line())
