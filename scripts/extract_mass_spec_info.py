#!/usr/bin/env python
"""Create total-ion-current images from a directory of ``.imzml`` files."""

from pathlib import Path

from argparse import ArgumentParser, Namespace

from numpy import save, log10

from pandas import read_csv

from skimage.util import img_as_ubyte
from skimage.io import imsave

from pyimzml.ImzMLParser import ImzMLParser  # type: ignore

from ispy_msi.file_utils import list_msi_data  # type: ignore
from ispy_msi.msi_extraction_utils import extract_img_data


def parse_command_line() -> Namespace:
    """Parse the command-line arguments.

    Returns
    -------
    Namespace
        Command-line arguments.

    """
    parser = ArgumentParser(description="Create total-ion-current images.")

    parser.add_argument(
        "msi_dir",
        type=Path,
        help="Directory holding corresponding '.imzl' and '.ibd'.",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Directory to save the total-ion-current images in",
        default="processed-data",
    )

    parser.add_argument(
        "--mc-csv",
        help="Path to the file with the mass-to-charge ratios.",
        type=Path,
        default="mass-to-charge-ratios.csv",
    )

    parser.add_argument(
        "--tol",
        help="Tolerance to use when extracting ion images.",
        type=float,
        default=7e-6,
    )

    return parser.parse_args()


def extract_imgs_from_single_scan(
    parser: ImzMLParser,
    imzml_path: Path,
    args: Namespace,
):
    """Extract the image-list data from each scan.

    Parameters
    ----------
    parser : ImzMLParser
        MSI parser  object.
    imzml_path : Path
        Path to the imzml file.
    args : Namespace
        Command-line arguments.

    """
    imgs = extract_img_data(parser, read_csv(args.mc_csv), args.tol)

    for img_type, img in imgs.items():
        save_path = args.out_dir / f"ion-imgs/{imzml_path.name}"
        save_path /= f"{img_type}.npy"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save(save_path, img)

    imsave(
        save_path.with_name("thumnail.png"),
        img_as_ubyte(log10(imgs["rms"].clip(1.0)) / log10(imgs["rms"].clip(1.0)).max()),
    )


def extract_from_single_scan(imzml_file: Path, args: Namespace):
    """Extract all data from a single scan.

    Parameters
    ----------
    imzml_file : Path
        Path to the scan in question.
    args : Namespace
        Command-line arguments.

    """
    parser = ImzMLParser(imzml_file)

    extract_imgs_from_single_scan(parser, imzml_file, args)


def extract_all_images(args: Namespace):
    """Create total-ion-current images from the user-requested files.

    Parameters
    ----------
    args : Namespace
        Command-line arguments.

    """
    data = list_msi_data(args.msi_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(data)

    for file_path in data.imzml:

        extract_from_single_scan(file_path, args)


if __name__ == "__main__":
    extract_all_images(parse_command_line())
