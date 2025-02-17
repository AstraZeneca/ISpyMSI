#!/usr/bin/env python
"""Record landmarks for sample-level image registration."""

from time import perf_counter

from typing import Tuple

from pathlib import Path

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace

from pandas import read_csv, DataFrame

from numpy import ndarray, load, percentile, concatenate

from tiffslide import TiffSlide


from skimage.exposure import equalize_adapthist  # pylint: disable=no-name-in-module
from skimage.transform import (  # pylint: disable=no-name-in-module
    AffineTransform,
    matrix_transform,
)


import matplotlib.pyplot as plt

from ispy_msi.geojson_utils import prepare_roi
from ispy_msi.mask_utils import multipolygon_to_numpy, edge_coords_from_polys


def _parse_command_line() -> Namespace:
    """Parse the command-line argument.

    Parameters
    ----------
    Namespace
        Command-line arguments.

    """
    parser = ArgumentParser(
        description="Visualise the landmark and mask-project process.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "base_dir",
        help="Base parent directory with all of the image data.",
        type=Path,
    )

    parser.add_argument(
        "--save-dir",
        help="Directory to save the output images in.",
        type=Path,
        default="landmark-visuals",
    )

    parser.add_argument(
        "--num-landmarks",
        type=int,
        help="Number of landmarks per sample to acquire.",
        default=5,
    )

    parser.add_argument(
        "--level",
        type=int,
        help="Level in the tiff to extract the H&E region from.",
        default=3,
    )

    parser.add_argument(
        "--metabolite",
        help="Ion image to use when recording the landmarks.",
        type=str,
        default="lactate",
    )

    parser.add_argument(
        "--landmark-csv",
        help="Location of the landmark csv file.",
        type=Path,
        default="landmarks.csv",
    )

    return parser.parse_args()


def _he_overview(
    he_path: Path,
    level: int,
) -> Tuple[ndarray, float]:
    """Return a low-power H&E overview image.

    Parameters
    ----------
    he_path : Path
        Path to the whole-slide image.
    level : int
        Level in the tiff to take the overview image at.

    Returns
    -------
    region : ndarray
        Overview of the H&E image.
    downscale : float
        The scale factor between the low-poer overview and the level-zero
        reference frame.

    Raises
    ------
    RuntimeError
        If we cannot extract a region from any level of the slide ``he_path``.

    """
    while level >= 0:
        try:
            with TiffSlide(he_path) as slide:
                dims = slide.level_dimensions[level]
                downscale = slide.level_downsamples[level]

                region = slide.read_region(
                    location=(0, 0),
                    level=level,
                    size=dims,
                    as_array=True,
                )
            return region, downscale
        except:  # pylint: disable=bare-except
            level -= 1

    raise RuntimeError(f"Failed to extract region on slide {he_path.name}.")


def single_msi_projection(
    msi_name: str,
    msi_frame: DataFrame,
    landmarks: DataFrame,
    args: Namespace,
):
    """Project the masks for a single slide.

    Parameters
    ----------
    msi_name : str
        Name of the MSI.
    msi_frame : DataFrame
        A data frame with the slide metadata for the msi in question.
    landmarks : DataFrame
        The landmarks file.
    args : Namespace
        Command-line arguments.

    """
    for row in msi_frame.itertuples():

        json_path = args.base_dir
        json_path /= f"ROI/{Path(row.he_img).with_suffix('.geojson')}"  # type: ignore

        msi_path = json_path.parent.with_name("ion-imgs")
        msi_path /= f"{msi_name}/{args.metabolite}.npy"

        msi = load(msi_path) / load(msi_path.with_name("rms.npy"))
        msi = msi.clip(0.0, percentile(msi, 99))
        msi = equalize_adapthist(msi / msi.max())

        object_df = prepare_roi(json_path)

        he_img, scale = _he_overview(
            json_path.parent.with_name("H&E") / row.he_img, args.level
        )

        produce_plot(he_img, msi, object_df, landmarks, scale, args.save_dir)


# pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
def produce_plot(
    he_img: ndarray,
    msi: ndarray,
    object_df: DataFrame,
    landmarks: DataFrame,
    scale: float,
    save_dir: Path,
):
    """Produce a plot of the landmarks on top of the samples.

    Parameters
    ----------
    he_img : ndarray
        H&E overview.
    msi : ndarray
        The mass spec. image.
    object_df : DataFrame
        The roi data for the sample of interest.
    landmarks : DataFrame
        The landmark csv.
    scale : float
        The downscaling applied to ``he_img``.
    save_dir : Path
        The directory to save the data in.

    """
    for row in object_df.itertuples():

        sample_marks = landmarks.loc[row.object_id == landmarks.roi]

        he_roi = multipolygon_to_numpy(row.roi)

        he_left, he_right, he_top, he_bottom = edge_coords_from_polys(he_roi)

        tfm = AffineTransform()
        tfm.estimate(
            sample_marks[["he_col", "he_row"]].to_numpy(),
            sample_marks[["msi_col", "msi_row"]].to_numpy(),
        )

        projected = []
        for coords in he_roi:
            projected.append(matrix_transform(coords, tfm.params))

        ms_left, ms_top = concatenate(projected, axis=0).min(axis=0)
        ms_right, ms_bottom = concatenate(projected, axis=0).max(axis=0)

        figure, axes = plt.subplots(1, 2)
        axes[0].imshow(he_img)  # type: ignore
        axes[1].imshow(msi, cmap="inferno")  # type: ignore

        axes[0].plot(  # type: ignore# type: ignore
            sample_marks.he_col / scale,
            sample_marks.he_row / scale,
            "kx",
            mew=1.5,
            ms=7,
        )

        axes[1].plot(  # type: ignore# type: ignore
            sample_marks.msi_col,
            sample_marks.msi_row,
            "wx",
            mew=1.5,
            ms=7,
        )

        axes[0].set_xlim(left=he_left / scale, right=he_right / scale)  # type: ignore
        axes[0].set_ylim(bottom=he_bottom / scale, top=he_top / scale)  # type: ignore

        axes[1].set_xlim(left=ms_left, right=ms_right)  # type: ignore
        axes[1].set_ylim(top=ms_top, bottom=ms_bottom)  # type: ignore

        for axis in axes.ravel():  # type: ignore
            axis.set_xticks([])
            axis.set_yticks([])

        figure.tight_layout(pad=0.01, w_pad=0.5)

        save_path = save_dir / f"{row.object_id}.svg"
        figure.savefig(save_path, dpi=250)

        for he_poly in he_roi:
            axes[0].plot(  # type: ignore
                he_poly[:, 0] / scale,
                he_poly[:, 1] / scale,
                "k-",
            )
        for msi_poly in projected:
            axes[1].plot(  # type: ignore
                msi_poly[:, 0],
                msi_poly[:, 1],
                "w-",
            )

        save_path = save_dir / f"{row.object_id}-with-roi.svg"
        figure.savefig(save_path, dpi=250)

        _figure_cleanup(axes)  # type: ignore


def _figure_cleanup(axes: ndarray):
    """Clean-up after matplotlib.

    Parameters
    ----------
    axes : ndarray
        Array of axes.

    """
    for axis in axes.ravel():
        axis.cla()
        axis.clear()
    plt.close("all")


def project_masks(args: Namespace):
    """Project the masks from the H&E space to that of the MSI.

    Paramaters
    ----------
    args : Namespace
        Command-line arguments

    """
    metadata = read_csv(args.base_dir / "metadata.csv")
    landmarks = read_csv(args.landmark_csv)

    json_dir = args.base_dir / "ROI"
    json_files = list(json_dir.glob("*.geojson"))
    he_stems = list(map(lambda x: x.stem, json_files))

    metadata["he_stem"] = metadata.he_img.apply(lambda x: Path(x).stem)

    metadata = metadata.loc[metadata.he_stem.isin(he_stems)]

    args.save_dir.mkdir(exist_ok=True, parents=True)

    print(metadata)

    print(f"There are {metadata.he_img.nunique()} H&E images.")
    print(f"There are {metadata.msi.nunique()} MSIs.")

    for msi_name, frame in metadata.groupby("msi"):

        start = perf_counter()

        single_msi_projection(str(msi_name), frame, landmarks, args)

        stop = perf_counter()

        print(f"Processed scan {msi_name} in {stop - start:.6f} seconds")


if __name__ == "__main__":
    project_masks(_parse_command_line())
