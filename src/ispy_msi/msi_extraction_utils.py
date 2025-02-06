"""Utility functions for creating total-ion-current images."""

from typing import Dict

from numpy import zeros, ndarray, array, square, sqrt, median
from numpy.linalg import norm

from scipy.stats import entropy  # type: ignore

from pandas import DataFrame

from pyimzml.ImzMLParser import ImzMLParser, getionimage  # type: ignore


def extract_img_data(
    parser: ImzMLParser,
    mass_to_charge: DataFrame,
    tol: float,
) -> Dict[str, ndarray]:
    """Extract all data of interest from the file at ``imzml_file``.

    Parameters
    ----------
    parser : ImzMLParser
        MSI parser object.
    mass_to_charge : DataFrame
        The mass to charge ratio for the metabolites of interest.
    tol : float
        The tolerence to use when selecting an ion image.

    Returns
    -------
    img_dict : Dict[str, ndarray]
        A dictionary of two-channel images.

    """
    img_dict = _get_norm_imgs(parser)
    img_dict.update(_get_ion_imgs(parser, mass_to_charge, tol))

    return img_dict


def _get_norm_imgs(parser: ImzMLParser) -> Dict[str, ndarray]:
    """Get the normalisation images.

    Parameters
    ----------
    parser : ImzMLParser
        The imzML parser.

    Returns
    -------
    imgs : Dict[str, ndarray]
        Normalisation images.

    """
    height = parser.imzmldict["max count of pixels y"]
    width = parser.imzmldict["max count of pixels x"]

    imgs = {
        key: zeros((height, width), dtype=float)
        for key in ["tic", "mic", "rms", "entropy", "median", "l2"]
    }

    for idx, (col, row, _) in enumerate(parser.coordinates):
        _, intensity = parser.getspectrum(idx)

        row, col = row - 1, col - 1

        intensity_arr = array(intensity)

        imgs["tic"][row, col] = intensity_arr.sum()
        imgs["mic"][row, col] = intensity_arr.mean()
        imgs["rms"][row, col] = sqrt(square(intensity_arr).mean())
        imgs["entropy"][row, col] = entropy(intensity_arr)
        imgs["median"][row, col] = median(intensity_arr)
        imgs["l2"][row, col] = norm(intensity_arr)

    return imgs


def _get_ion_imgs(
    parser: ImzMLParser,
    mass_to_charge: DataFrame,
    tol: float,
) -> Dict[str, ndarray]:
    """Extract the ion images given in ``mass_to_charge``.

    Parameters
    ----------
    parser : ImzMLParser
        The MSI object.
    mass_to_charge : DataFrame
        The data frame with the metabolite metadata.
    tol : float
        The tolerance to use when extracting the images.

    Parameters
    ----------
    img_dict[str, ndarray]
        Dictionary of ion images.

    """
    img_dict: Dict[str, ndarray] = {}
    for row in mass_to_charge.itertuples():
        img_dict[str(row.name).lower()] = getionimage(  # tpye: ignore
            parser,
            row.mass_to_charge,  # type: ignore
            tol=tol * row.mass_to_charge,  # type: ignore
        )

    return img_dict
