#!/usr/bin/env python
"""Analyse the image statistics."""
from pathlib import Path
from argparse import Namespace, ArgumentParser, ArgumentDefaultsHelpFormatter

from numpy import load, ceil, floor, arange  # pylint: disable=no-name-in-module

from pandas import DataFrame, read_csv

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar  # type: ignore

import ispy_msi
from ispy_msi.patch_extraction_utils import (
    preprocess_tic_img,
    preprocess_rms_img,
    preprocess_entropy_img,
)

plt.style.use(Path(ispy_msi.__file__).with_name("matplotlibrc"))


def _parse_command_line():
    """Parser the command-line arguments.

    Returns
    -------
    Namespace
        Command-line arguments in a Namespace.

    """
    parser = ArgumentParser(
        description="Image representation analysis.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "image_dir",
        help="Directory holding the images.",
        type=Path,
    )

    parser.add_argument(
        "--save-dir",
        help="Directory to save the plots in.",
        type=Path,
        default="representation-visuals",
    )

    return parser.parse_args()


def analyse_images(args: Namespace):  # pylint: disable=too-many-locals
    """Analyse the images.

    Parameters
    ----------
    args : Namespace
        Command-line arguments.

    """
    metadata = read_csv(args.image_dir.parent / "metadata.csv")
    res = metadata.drop_duplicates(subset="msi")[["msi", "msi_microns_pp"]]
    res = res.set_index("msi").msi_microns_pp  # type: ignore

    images = DataFrame()

    args.save_dir.mkdir(parents=True, exist_ok=True)

    styles = ["tic", "rms", "entropy"]

    for style in styles:
        images[style] = list(args.image_dir.glob(f"*/{style}.npy"))

    for row in images.itertuples():

        mpp = res[row.tic.parent.name]  # type: ignore

        tic = preprocess_tic_img(load(row.tic))  # type: ignore
        rms = preprocess_rms_img(load(row.rms))  # type: ignore
        entropy = preprocess_entropy_img(load(row.entropy))  # type: ignore

        height, width = tic.shape
        fig_width = 8.27
        fig_height = fig_width / (3 * (width / height))
        fig_height *= 0.9

        figure, axes = plt.subplots(
            1,
            3,
            figsize=(fig_width, fig_height + 0.25),
        )

        for img, axis in zip([tic, rms, entropy], axes.ravel()):  # type: ignore

            im = axis.imshow(
                img,
                cmap="inferno",
                vmin=floor(img.min()),
                vmax=ceil(img.max()),
            )

            divider = make_axes_locatable(axis)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax, pad=0.0)

            cbar.set_ticks(
                arange(floor(img.min()), ceil(img.max()) + 1, 1),  # type: ignore
            )
            # cbar.set_ticklabels(["Min", "Max"], size=7)

            scalebar = AnchoredSizeBar(
                axis.transData,
                5000 / mpp,
                r"$5\ \mathrm{mm}$",
                "lower right",
                color="white",
                frameon=False,
            )

            axis.add_artist(scalebar)

        labels = ["(a) — TIC", "(b) — RMS", "(c) — Entropy"]
        for axis, label in zip(axes.ravel(), labels):  # type: ignore
            axis.set_xticks([])
            axis.set_yticks([])
            axis.text(
                0.05,
                0.90,
                label,
                transform=axis.transAxes,
                color="white",
                fontsize=12,
            )

        figure.tight_layout(pad=0.0, w_pad=0.25)

        figure.savefig(
            args.save_dir / row.tic.parent.with_suffix(".pdf").name,  # type: ignore
            dpi=500,
        )
        plt.close("all")


if __name__ == "__main__":
    analyse_images(_parse_command_line())
