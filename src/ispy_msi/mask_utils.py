"""Mask-specific utility functions."""

from typing import Tuple, List

from numpy import ndarray, zeros, asarray, concatenate, where

from shapely import MultiPolygon  # type: ignore

from skimage.draw import polygon2mask  # type: ignore # pylint: disable=no-name-in-module


def edge_coords_from_polys(polys: List[ndarray]) -> Tuple[int, int, int, int]:
    """Return the edge coordinates from the polygons.

    Parameters
    ----------
    polys : List[ndarray]
        List of the polygons. Each array should be of the form
        ``[(col, row), ...]``.

    Returns
    -------
    left : int
        The left-most coordinate in the polygons.
    right : int
        The right-most coordinate in the polygons.
    top : int
        The top-most coordinate in the polygons.
    bottom : int
        The bottom-most coordinate in the polygons.

    """
    all_polys = concatenate(polys, axis=0)

    right, bottom = all_polys.max(axis=0)
    left, top = all_polys.min(axis=0)

    return left, right, top, bottom


def edge_coords_from_binary_mask(binary_mask: ndarray) -> Tuple[int, int, int, int]:
    """Return the edge coordinates in ``binary_mask``.

    Parameters
    ----------
    binary_mask : ndarray
        The binary mask.

    Returns
    -------
    left : int
        The left-most coordinate in the mask.
    right : int
        The right-most coordinate in the mask.
    top : int
        The top-most coordinate in the mask.
    bottom:
        The bottom-most coordinate in the mask.

    """
    rows, cols = where(binary_mask)  # pylint: disable=unbalanced-tuple-unpacking

    left, right = cols.min(), cols.max()
    top, bottom = rows.min(), rows.max()

    return left, right, top, bottom


def multipolygon_to_numpy(roi: MultiPolygon) -> List[ndarray]:
    """Convert the ``MultiPolygon`` object to numpy arrays.

    Parameters
    ----------
    roi : MultiPolygon
        The region(s) of interest.


    Returns
    -------
    numpy_polys : List[ndarray]
        The boundary coordinates of each ROI as a numpy array.

    """
    numpy_polys = []
    for poly in roi.geoms:
        numpy_polys.append(asarray(poly.exterior.coords))

    return numpy_polys


def mask_from_polys(polys: List[ndarray], height: int, width: int) -> ndarray:
    """Extract a binary mask from the polygons in ``polys``.

    Parameters
    ----------
    polys : List[ndarray]
        List of polyons delineating the regions of interest:
        ``[(col, row), ...]``.
    height : int
        The height of the output image.
    width : int
        The width of the output image.

    Returns
    -------
    mask : ndarray
        A binary mask demarcating the regions of interest.

    """
    mask = zeros((height, width), dtype=bool)
    for poly in polys:
        # Flipping so we have [(row, col), ...] and not [(col, row), ...]
        poly = poly[:, ::-1]
        mask = polygon2mask((height, width), poly) | mask

    return mask
