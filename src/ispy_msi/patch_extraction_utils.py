"""Patch-extraction utilities."""

from typing import List, Tuple, Dict
from pathlib import Path
from itertools import product

from skimage.exposure import equalize_adapthist  # pylint: disable=no-name-in-module
from skimage.io import imread, imsave  # type: ignore
from skimage.util import img_as_ubyte


from numpy import load, ndarray, log10, concatenate, where, percentile, array, pad


def _percentile_normalise(
    grey_img: ndarray,
    low_perc: float = 1.0,
    high_perc: float = 99.0,
) -> ndarray:
    """Normalise ``grey_img`` to the range [0, 1].

    Parameters
    ----------
    grey_img : ndarray
        A single-channel image of any float values.

    Returns
    -------
    ndarray
        ``grey_img`` normalised to [0, 1]

    """
    low, high = percentile(grey_img, (low_perc, high_perc))
    return ((grey_img - low) / (high - low)).clip(0.0, 1.0)


def _coord_iter(img_height: int, img_width: int, stride: int) -> ndarray:
    """Return an iterator across the image's coordinates.

    Parameters
    ----------
    img_height : int
        The height of the image to be tesselated.
    img_width : int
        The width of the image to be tesselated.
    stride : int
        Stride to use in the patch extraction.

    Returns
    -------
    ndarray
        Array of coordinates to iterate over.

    """
    rows = range(0, img_height, stride)
    cols = range(0, img_width, stride)
    return array(list(product(rows, cols)))


def _patch_edges(
    top: int,
    left: int,
    height: int,
    width: int,
    patch_size: int,
) -> Tuple[int, int, int, int]:
    """Correct the patch coordinates if they go outside the image dims.

    Parameters
    ----------
    top : int
        The row (top) coordinate of the patch.
    left : int
        The col (left) coord of the patch.
    height : int
        The height of the image.
    width : int
        The width of the image.
    patch_size : int
        The size of the patches we are extracting.

    Returns
    -------
    Tuple[int, int, int, int]
        The left, right, top and bottom patch coords.

    Raises
    ------
    ValueError
        If ``patch_size`` is less than 2.
    ValueError
        If the left coordinate is outside the image's horizontal range.

    """
    if patch_size < 2:
        raise ValueError(f"Patch should be at least 2, got '{patch_size}'.")

    right = left + patch_size
    bottom = top + patch_size

    if (diff := right - width) > 0:
        left, right = left - diff, right - diff

    if (diff := bottom - height) > 0:
        top, bottom = top - diff, bottom - diff

    if not 0 <= left < width:
        msg = f"Left coord '{left}' outside horizontal range [0, {width}]."
        raise ValueError(msg)

    if not 0 <= right <= width:
        msg = f"Right coord '{right}' outside horizontal range [0, {width}]."
        raise ValueError(msg)

    if not 0 <= top < height:
        msg = f"Top coord '{top}' outside vertical range [0, {height}]."
        raise ValueError(msg)

    return left, right, top, bottom


def get_three_channel_msi_rep(
    tic_path: Path,
    rms_path: Path,
    entropy_path: Path,
) -> ndarray:
    """Get a three-channel image representation.

    Parameters
    ----------
    tic_path : Path
        Total-ion-current image path.
    rms_path : Path
        Root-mean-square image path.
    entropy_path : Path
        Entropy image path.

    Returns
    -------
    ndarray
        Three-channel image.

    """
    single_imgs: List[ndarray] = []

    single_imgs.append(preprocess_tic_img(load(tic_path)))
    single_imgs.append(preprocess_rms_img(load(rms_path)))
    single_imgs.append(preprocess_entropy_img(load(entropy_path)))

    return concatenate(list(map(lambda x: x[:, :, None], single_imgs)), axis=2)


def preprocess_tic_img(tic_img: ndarray) -> ndarray:
    """Preprocess the ``tic_img``.

    Parameters
    ----------
    tic_img : ndarray
        Mean-ion-current image.

    Returns
    -------
    ndarray
        The preprocessed image

    """
    tic_img = log10(1 + tic_img)
    tic_img = _percentile_normalise(tic_img)

    return equalize_adapthist(tic_img)


def preprocess_rms_img(rms_img: ndarray) -> ndarray:
    """Preprocess the ``rms_img``.

    Parameters
    ----------
    rms_img : ndarray
        Root-mean-square image.

    Returns
    -------
    ndarray
        The preprocessed image.

    """
    return preprocess_tic_img(rms_img)


def preprocess_entropy_img(entropy_img: ndarray) -> ndarray:
    """Preprocess the ``entropy_img``.

    Parameters
    ----------
    entropy_img : ndarray
        Entropy image.

    Returns
    -------
    entropy_img : ndarray
        The preprocessed image

    """
    return _percentile_normalise(entropy_img)


def _tessellate_single_img(  # pylint: disable=too-many-locals, too-many-arguments
    parent_img: ndarray,
    patch_size: int,
    stride: int,
    save_dir: Path,
    msi_name: str,
):
    """Tessellate a single image.

    Parameters
    ----------
    parent_img : ndarray
        The image to be tiled.
    patch_size : int
        Length of the square patches.
    stride : int
        Stride to use in the sliding-window patch generation.
    save_dir : Path
        Directory to save the patches in.
    msi_name : str
        The name of the mass spec. image.

    """
    save_dir.mkdir(exist_ok=True, parents=True)

    parent_img = _pad_image(
        image=parent_img,
        patch_size=patch_size,
        stride=stride,
    )

    height, width = parent_img.shape[0], parent_img.shape[1]

    for row, col in _coord_iter(height, width, stride):
        left, right, top, bottom = _patch_edges(
            row,
            col,
            height,
            width,
            patch_size,
        )

        coord_str = f"[x={col},y={row},w={patch_size},h={patch_size}]"

        patch = img_as_ubyte(parent_img[top:bottom, left:right])
        imsave(
            save_dir / f"{msi_name}---{coord_str}.png",
            patch,
            check_contrast=False,
        )


def mask_bbox_coords(mask: ndarray) -> Dict[str, int]:
    """Get the coordinates defining a box drawn round the entire mask.

    Parameters
    ----------
    mask : ndarray
        Binary tissue mask for the MSI.

    Returns
    -------
    Dict[str, int]
        Dictionary holding the box coords.

    """
    rows, cols = where(mask.clip(0, 1))  # pylint: disable=unbalanced-tuple-unpacking
    return {
        "top": rows.min(),
        "bottom": rows.max(),
        "left": cols.min(),
        "right": cols.max(),
    }


def _pad_image(image: ndarray, patch_size: int, stride: int):
    """Pad the image so it can be tiled into a regular grid.

    Parameters
    ----------
    image : ndarray
        The image to be padded, or not.
    patch_size : int
        The size of the patches to extract.
    stride : int
        The stride to be used in the patch extraction.

    Returns
    -------
    ndarray
        Padded version of ``image``.

    """
    if image.ndim == 2:
        image = image[:, :, None]

    height, width, _ = image.shape

    max_row, max_col = _coord_iter(height, width, stride).max(axis=0)
    max_top = max_row + patch_size
    max_right = max_col + patch_size

    height_pad = max(max_top - height, 0)
    width_path = max(max_right - width, 0)

    return pad(
        image, pad_width=((0, height_pad), (0, width_path), (0, 0)), constant_values=0
    ).squeeze()


def tessellate_image_mask_pair(  # pylint: disable=too-many-arguments
    save_dir: Path,
    msi_rep_path: Path,
    mask_path: Path,
    patch_size: int,
    stride: int,
):
    """Save the patches to file.

    Parameters
    ----------
    save_dir : Path
        Directory to save the patches in.
    tic_path : Path
        Path to the TIC image.
    msi_rep : ndarray
        Three-channel MSI representation.
    mask_path : Path
        Path to the mask image.
    patch_size : int
        The length of the square patches to be generated.

    """
    msi_name = msi_rep_path.stem
    msi_rep = imread(msi_rep_path)
    mask = imread(mask_path)

    _tessellate_single_img(
        msi_rep,
        patch_size,
        stride,
        save_dir / "patches",
        msi_name,
    )

    _tessellate_single_img(
        mask,
        patch_size,
        stride,
        save_dir / "patch-masks",
        msi_name,
    )
