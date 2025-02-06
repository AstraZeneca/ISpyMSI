"""Affine-transform-based mask transfer using landmarks."""

from typing import List, Optional, Tuple


from numpy import ndarray, ones


import matplotlib.pyplot as plt  # type: ignore

from skimage.exposure import equalize_adapthist  # pylint: disable=no-name-in-module
from skimage.filters import median  # pylint: disable=no-name-in-module


class LandMarker:  # pylint: disable=too-few-public-methods
    """Landmark acquisition object.

    Parameters
    ----------
    num_landmarks : int, optional
        The number of landmarks to request.

    """

    def __init__(self, num_landmarks: int = 5):
        """Build ``LandmarkReg``."""
        self._num_landmarks = num_landmarks

    def _get_landmarks(
        self,
        src_img: ndarray,
        dst_img: ndarray,
        roi: Optional[List[ndarray]] = None,
    ):
        """Get the corresponding landmarks between the two images.

        Parameters
        ----------
        src_img : ndarray
            The source (annotated image).
        dst_img : ndarray
            The image whose space we wish to map to.
        roi : List[ndarray], optional
            Region of interest to highlight in ``src_img``.

        """
        src_coords = ones((self._num_landmarks, 2))
        dst_coords = ones((self._num_landmarks, 2))

        dst_img = median(dst_img, ones((3, 3)))

        dst_img -= dst_img.min()
        dst_img /= dst_img.max()

        dst_img = equalize_adapthist(dst_img)

        figure, axes = plt.subplots(1, 2)
        axes[0].imshow(src_img)  # type: ignore
        axes[1].imshow(dst_img, cmap="inferno")  # type: ignore

        for region in roi if roi is not None else []:
            axes[0].plot(region[:, 0], region[:, 1], "-b")  # type: ignore

        for idx in range(self._num_landmarks):
            for coords, axis in zip([src_coords, dst_coords], axes.ravel()):  # type: ignore
                axis.set_title("Spacebar here")

                point = figure.ginput(1, mouse_add=None, timeout=0)[0]  # type: ignore
                coords[idx, :2] = point

                axis.plot(point[0], point[1], "go")

                plt.draw()

                axis.set_title("")

        plt.close(figure)

        return src_coords, dst_coords

    def acquire(
        self,
        src_img: ndarray,
        dst_img: ndarray,
        roi: Optional[List[ndarray]] = None,
    ) -> Tuple[ndarray, ndarray]:
        """Transfer the masks using the two reference images.

        Parameters
        ----------
        src_img : ndarray
            The source image, which has been annotated.
        dst_img : ndarray
            The image whose space we wish to map to.
        roi : List[ndarray], optional
            Regions of interest to highlight ``src_img`` as helpful reference
            for the user.

        Returns
        -------
        src_coords : ndarray
            Landmark coordinates in ``src_img``.
        dst_coords : ndarray
            Landmakrd coordinates in ``dst_img``.

        """
        src_coords, dst_coords = self._get_landmarks(src_img, dst_img, roi)
        return src_coords, dst_coords
