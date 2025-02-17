#!/usr/bin/env python
"""Record landmarks for sample-level image registration."""

from time import perf_counter

from typing import Tuple

from pathlib import Path

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from argparse import BooleanOptionalAction

from pandas import read_csv, DataFrame

from numpy import ndarray, zeros

from tiffslide import TiffSlide

import matplotlib.pyplot as plt

from skimage.transform import (  # pylint: disable=no-name-in-module
    AffineTransform,
    matrix_transform,
)
from skimage.draw import polygon2mask  # pylint: disable=no-name-in-module
from skimage.util import img_as_ubyte
from skimage.io import imsave


from ispy_msi.geojson_utils import prepare_roi
from ispy_msi.mask_utils import multipolygon_to_numpy
from ispy_msi.patch_extraction_utils import get_three_channel_msi_rep, mask_bbox_coords

from ispy_msi.plotting import figure_cleanup


def _parse_command_line() -> Namespace:
    """Parse the command-line argument.

    Parameters
    ----------
    Namespace
        Command-line arguments.

    """
    parser = ArgumentParser(
        description="Extract landmarks for image registration.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "base_dir",
        help="Base parent directory with all of the image data.",
        type=Path,
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
        "--landmark-csv",
        help="Location of the landmark csv file.",
        type=Path,
        default="landmarks.csv",
    )

    parser.add_argument(
        "--test-only",
        help="Whether to only use the test images or not.",
        type=bool,
        default=False,
        action=BooleanOptionalAction,
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


def _single_msi_projection(  # pylint: disable=too-many-locals
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

    Notes
    -----
    When saving the MSI representations and masks, we sometimes have examples
    where bits of the tissue are still unannotated, because of weird cropping
    during scanning. To deal with this, we crop all images and masks so the
    mask's egdes become the edges of the image.

    """
    save_dir = args.base_dir / "images-and-masks"
    msi_path = args.base_dir / f"ion-imgs/{msi_name}"

    msi = img_as_ubyte(
        get_three_channel_msi_rep(
            msi_path / "tic.npy",
            msi_path / "rms.npy",
            msi_path / "entropy.npy",
        )
    )

    mask = zeros((msi.shape[0], msi.shape[1])).astype(bool)

    for row in msi_frame.itertuples():

        json_path = args.base_dir
        json_path /= f"ROI/{Path(row.he_img).with_suffix('.geojson')}"  # type: ignore

        msi_path = json_path.parent.with_name("ion-imgs") / msi_name

        object_df = prepare_roi(json_path)

        he_img, scale = _he_overview(
            json_path.parent.with_name("H&E") / row.he_img, args.level
        )

        base_path = save_dir / f"{row.split}"

        save_path = base_path / f"visuals/{row.he_img}.pdf"
        _plot_projected_masks(he_img, scale, msi, landmarks, object_df, save_path)

        mask = _update_mask_with_one_slide(mask, landmarks, object_df)

    box = mask_bbox_coords(mask)

    _save_image_rep(
        msi[box["top"] : box["bottom"], box["left"] : box["right"]],
        base_path / f"images/{msi_name}.png",
    )
    _save_mask(
        mask[box["top"] : box["bottom"], box["left"] : box["right"]],
        base_path / f"masks/{msi_name}.png",
    )


def _plot_projected_masks(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    he_img: ndarray,
    scale: float,
    msi: ndarray,
    landmarks: DataFrame,
    object_df: DataFrame,
    save_path: Path,
):
    """Visualise the projected masks.

    Parameters
    ----------
    he_img : ndarray
        H&E overview image.
    scale : float
        The scale factor separating the H&E overview image from its
        level-zero reference frame.
    msi : ndarray
        The mass spec. image.
    landmarks : DataFrame
        The metadata holding the landamarks for each object.
    object_df : DataFrame
        Data frame holding all the object-level ROI data.
    save_path : Path
        Path to save the figure object to.

    """
    figure, axes = plt.subplots(1, 2)
    axes[0].imshow(he_img)  # type: ignore
    axes[1].imshow(msi)  # type: ignore

    for row in object_df.itertuples():

        roi = multipolygon_to_numpy(row.roi)

        landmark_slice = landmarks.loc[landmarks.roi == row.object_id]

        tfm = AffineTransform()
        tfm.estimate(
            landmark_slice[["he_col", "he_row"]].to_numpy(),
            landmark_slice[["msi_col", "msi_row"]].to_numpy(),
        )

        for coords in roi:

            msi_coords = matrix_transform(coords, tfm.params)

            axes[0].plot(  # type: ignore
                coords[:, 0] / scale,
                coords[:, 1] / scale,
                "magenta",
                lw=0.5,
            )
            axes[1].plot(  # type: ignore
                msi_coords[:, 0],
                msi_coords[:, 1],
                "magenta",
                lw=0.5,
            )

    for axis in axes.ravel():  # type: ignore
        axis.set_xticks([])
        axis.set_yticks([])

    save_path.parent.mkdir(exist_ok=True, parents=True)

    figure.tight_layout(pad=0.05)
    figure.savefig(save_path, dpi=500)

    figure_cleanup(axes)  # type: ignore


def _update_mask_with_one_slide(
    mask: ndarray,
    landmarks: DataFrame,
    object_df: DataFrame,
) -> ndarray:
    """Visualise the projected masks.

    Parameters
    ----------
    mask : ndarray
        The tissue mask, to be completed.
    landmarks : DataFrame
        The metadata holding the landamarks for each object.
    object_df : DataFrame
        Data frame holding all the object-level ROI data.

    Returns
    -------
    mask : ndarray
        A version of ``mask`` with objects added.

    """
    for row in object_df.itertuples():

        roi = multipolygon_to_numpy(row.roi)

        landmark_slice = landmarks.loc[landmarks.roi == row.object_id]

        tfm = AffineTransform()
        tfm.estimate(
            landmark_slice[["he_col", "he_row"]].to_numpy(),
            landmark_slice[["msi_col", "msi_row"]].to_numpy(),
        )

        for coords in roi:

            msi_coords = matrix_transform(coords, tfm.params)

            mask = mask | polygon2mask(mask.shape, msi_coords[:, ::-1])

    return mask


def _save_image_rep(img_rep: ndarray, save_path: Path):
    """Save the image representation.

    Parameters
    ----------
    img_rep : ndarray
        The image representation.
    save_path : Path
        The target file path.

    """
    save_path.parent.mkdir(exist_ok=True, parents=True)
    imsave(save_path, img_rep)


def _save_mask(mask: ndarray, save_path: Path):
    """Save the mask to file.

    Parameters
    ----------
    mask : ndarray
        The boolean mask.
    save_path : Path
        The target file path.

    """
    save_path.parent.mkdir(exist_ok=True, parents=True)
    imsave(save_path, img_as_ubyte(mask))


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

    print(metadata)

    print(f"There are {metadata.he_img.nunique()} H&E images.")
    print(f"There are {metadata.msi.nunique()} MSIs.")

    for msi_name, frame in metadata.groupby("msi"):

        start = perf_counter()

        is_training = frame.split.unique().item() == "train"
        if (args.test_only is True) and is_training:
            continue

        _single_msi_projection(str(msi_name), frame, landmarks, args)

        stop = perf_counter()

        print(f"Processed scan {msi_name} in {stop - start:.6f} seconds")


if __name__ == "__main__":
    project_masks(_parse_command_line())
