"""Custom metrics."""

import torch
from torch import Tensor, no_grad

from numpy import nan


@no_grad()
def dice(pred: Tensor, target: Tensor) -> float:
    """Compute the dice coefficient between ``pred`` and ``target``.

    Parameters
    ----------
    pred : Tensor
        Tensor holding a model's predictions (should be boolean).
    target : Tensor
        The corresponding target for ``pred`` (should be boolean).

    Returns
    -------
    dice: float
        The dice score.

    Notes
    -----
    - This function should work with inputs of any shape, so long as they match
    and the tensors are boolean.
    - If ``pred`` and ``target`` are all negative, this function returns
    ``np.nan``.

    """
    _is_bool_tensor(pred)
    _is_bool_tensor(target)

    if not pred.shape == target.shape:
        msg = "Pred and target shapes should match. "
        msg += f"Got '{pred.shape}' and '{target.shape}'."
        raise RuntimeError(msg)

    numerator = 2.0 * (pred & target).float().sum().item()
    denominator = (pred.sum() + target.sum()).item()

    try:
        dice_score = numerator / denominator
    except ZeroDivisionError:
        dice_score = nan

    return dice_score


def _is_bool_tensor(bool_tensor: Tensor):
    """Make sure ``bool_tensor`` is a boolean ``Tensor``.

    Parameters
    ----------
    bool_tensor : Tensor
        The tensor to check.

    Raises
    ------
    TypeError
        If ``bool_tensor`` is not a ``Tensor``.
    TypeError
        If ``bool_tensor``'s dtype is not ``torch.bool``.


    """
    if not isinstance(bool_tensor, Tensor):
        raise TypeError(f"Expected 'Tensor'. Got '{type(bool_tensor)}'.")

    if not bool_tensor.dtype == torch.bool:  # pylint: disable=no-member
        raise TypeError(f"Expected boolean Tensor, got '{bool_tensor.dtype}'.")
