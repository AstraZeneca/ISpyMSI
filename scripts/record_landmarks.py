#!/usr/bin/env python
"""Record landmarks for sample-level image registration."""

from typing import List, Tuple

from pathlib import Path

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace

from pandas import read_csv, DataFrame

from numpy import ndarray, concatenate, load, percentile

from tiffslide import TiffSlide

from skimage.segmentation import mark_boundaries
from skimage.exposure import equalize_adapthist  # pylint: disable=no-name-in-module


from ispy_msi.landmark_utils import LandMarker
from ispy_msi.geojson_utils import prepare_roi
from ispy_msi.mask_utils import multipolygon_to_numpy, mask_from_polys


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
        "--metabolite",
        help="Ion image to use when recording the landmarks.",
        type=str,
        default="lactate",
    )

    parser.add_argument(
        "--landmark-csv",
        help="File to save the landmark data in.",
        type=Path,
        default="landmarks.csv",
    )

    return parser.parse_args()


def _he_overview(
    he_path: Path,
    polys: List[ndarray],
    level: int,
) -> Tuple[ndarray, float]:
    """Return a low-power H&E overview image.

    Parameters
    ----------
    he_path : Path
        Path to the whole-slide image.
    polys : List[ndarray]
        List of polygons delineating an object of interest.
    level : int
        Level in the tiff to take the overview image at.

    Returns
    -------
    region : ndarray
        Overview of the H&E image.
    downscale : float
        The scale factor between the low-poer overview and the level-zero
        reference frame.

    """
    with TiffSlide(he_path) as slide:
        dims = slide.level_dimensions[level]
        downscale = slide.level_downsamples[level]

        region = slide.read_region(
            location=(0, 0),
            level=level,
            size=dims,
            as_array=True,
        )

    rescaled = [poly.copy() / downscale for poly in polys]

    mask = mask_from_polys(rescaled, region.shape[0], region.shape[1])

    return mark_boundaries(region, mask, color=(0, 0, 0)), downscale


def _exist_check(
    landmark_path: Path,
    he_name: str,
    msi_name: str,
    roi: str,
) -> bool:
    """Check if the landmarks have already been recorded for the object.

    Parameters
    ----------
    landmark_path : Path
        The path to the landmarks file.
    he_name : str
        Name of the H&E image,
    msi_name : str
        Name of the MSI.
    roi : str
        Unique ID of the region of interest.

    Returns
    -------
    bool
        ``True`` if the landmarks have already been recorded, else ``False``.

    """
    if not landmark_path.exists():
        return False

    landmarks = read_csv(landmark_path)

    return all(
        [
            he_name in landmarks["he_name"].to_list(),
            msi_name in landmarks["msi_name"].to_list(),
            roi in landmarks["roi"].to_list(),
        ]
    )


def _save_landmarks(landmarks: DataFrame, target_path: Path):
    """Save ``landmarks`` to file.

    Parameters
    ----------
    landmarks : DataFrame
        The landmark data to be recorded.
    target_path : Path
        The target file path.

    """
    if not target_path.parent.exists():
        target_path.parent.mkdir(exist_ok=True, parents=True)

    if not target_path.exists():
        landmarks.to_csv(target_path, index=False)
    else:
        landmarks.to_csv(target_path, mode="a", header=False, index=False)


def record_for_single_slide(
    json_path: Path,
    he_name: str,
    msi_name: str,
    args: Namespace,
):
    """Record the landmarks for a single histological slide.

    Parameters
    ----------
    json_path : path
        Path to the json file with the annotations.
    he_name : str
        Name of the histological image file.
    msi_name : str
        Name of the MSI.
    args : Namespace
        Command-line arguments.

    """
    marker = LandMarker(num_landmarks=args.num_landmarks)
    object_df = prepare_roi(json_path)

    for row in object_df.itertuples():

        roi = multipolygon_to_numpy(row.roi)
        msi_path = json_path.parent.with_name("ion-imgs")
        msi_path /= f"{msi_name}/{args.metabolite}.npy"

        if _exist_check(args.landmark_csv, he_name, msi_name, row.object_id):  # type: ignore
            continue

        norm_img = load(msi_path.with_name("rms.npy"))
        norm_img = norm_img.clip(*percentile(norm_img, (0.1, 100.0)))

        msi = load(msi_path) / norm_img
        msi = msi.clip(0.0, percentile(msi, 99))
        msi = equalize_adapthist(msi / msi.max())

        he_img, scale = _he_overview(
            json_path.parent.with_name("H&E") / he_name,
            roi,
            args.level,
        )

        landmarks = DataFrame(
            columns=["he_col", "he_row", "msi_col", "msi_row"],
            data=concatenate(marker.acquire(he_img, msi), axis=1),
        )
        landmarks[["he_col", "he_row"]] *= scale

        landmarks["he_name"] = he_name
        landmarks["msi_name"] = msi_name
        landmarks["roi"] = row.object_id

        _save_landmarks(landmarks, args.landmark_csv)


def record_landmarks(args: Namespace):
    """Record the landmarks for each sample.

    Paramaters
    ----------
    args : Namespace
        Command-line arguments

    """
    metadata = read_csv(args.base_dir / "metadata.csv")

    json_dir = args.base_dir / "ROI"
    json_files = list(json_dir.glob("*.geojson"))

    for json_file in json_files:

        he_name = str(Path(json_file.name).with_suffix(".svs"))
        print(he_name, metadata.loc[metadata.he_img == he_name, "msi"].item())

        msi_name = metadata.loc[metadata.he_img == he_name, "msi"].item()
        print(json_file, msi_name)

        record_for_single_slide(json_file, he_name, msi_name, args)


if __name__ == "__main__":
    record_landmarks(_parse_command_line())
