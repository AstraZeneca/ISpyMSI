"""Training and validation transforms."""

from typing import List, Callable, Union

from torch_tools.torch_utils import target_from_mask_img

from torch import (  # pylint: disable=no-name-in-module
    Tensor,
    from_numpy,
    randint,
    randn_like,
)


from torchvision.transforms import (  # type: ignore
    Compose,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomInvert,
)

from torchvision.transforms.functional import to_tensor  # type: ignore
from torchvision.transforms.functional import rotate  # type: ignore

from skimage.io import imread

from numpy import ndarray


def gaussian_noise(img_tensor: Tensor, var: float = 1e-4) -> Tensor:
    """Speckle ``img_tensor`` with Gaussian noise.

    Parameters
    ----------
    img_tensor : Tensor
        The image to be noised.
    var : float, optional
        The variance of the noise.

    Returns
    -------
    Tensor
        A noisy version of ``img_tensor``

    """
    std_dev = var**0.5

    noise = randn_like(img_tensor) * std_dev

    return img_tensor + noise


def as_tensor(image_array: ndarray) -> Tensor:
    """Prepare ``image_array`` as a Tensor.

    Parameters
    ----------
    image_array
        The image as a numpy array.

    Returns
    -------
    ``image_array`` as a tensor.

    """
    return to_tensor(image_array).float()


def _binary_mask(grey_tensor: Tensor):
    """Create a target tensor for a grey mask.

    Parameters
    ----------
    grey_tensor : Tensor
        A 2D ``Tensor``, where each pixel value is the class index.

    Returns
    -------
    Tensor
        Target tensor for shape (1, 2, H, W).

    """
    grey_tensor = grey_tensor.clip(0, 1)
    return target_from_mask_img(grey_tensor, 2)


def _on_axis_rotation(image: Tensor) -> Tensor:
    """Rotate image by a random integer multiple of 90 degrees.

    Parameters
    ----------
    image : Tensor
        The image to rotate.

    Returns
    -------
    Tensor
        The rotated image.

    """
    angle = randint(4, (1,)).item() * 90.0
    return rotate(image, angle)


def compose_input_tfms(training: bool = False) -> Compose:
    """Return a composition of transforms.

    Parameters
    ----------
    training : bool, optional
        Whether the transforms are for training or validation.

    Returns
    -------
    Compose
        Composition of transforms.

    """
    transform_list: List[Callable] = [imread, as_tensor]

    # After tensor augmentation
    if training is True:
        transform_list += [gaussian_noise, RandomInvert()]

    return Compose(transform_list)


def compose_target_tfms() -> Compose:
    """Return a composition of target transforms.

    Returns
    -------
    Compose
        Composition of transforms.

    """
    return Compose([imread, from_numpy, _binary_mask])


def compose_both_tfms(training: bool) -> Union[Compose, None]:
    """Compose target transforms.

    Parameters
    ----------
    training : bool
        Are we training, or validating?

    """
    if training is True:
        return Compose(
            [
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                _on_axis_rotation,
            ]
        )

    return None
