"""Tests for ``ispy_msi.metrics``."""

import pytest

from torch import tensor  # pylint: disable=no-name-in-module
from numpy import isnan

from ispy_msi import metrics


def test_dice_input_types():
    """Test the types accepted by the arguments to ``metrics.dice``."""
    # Should work with boolean tensors
    _ = metrics.dice(
        tensor([0, 1, 0], dtype=bool),
        tensor([0, 1, 0], dtype=bool),
    )

    # Should break with non-Tensor
    with pytest.raises(TypeError):
        _ = metrics.dice([True, False], tensor([0, 1, 0], dtype=bool))

    with pytest.raises(TypeError):
        _ = metrics.dice(tensor([0, 1, 0], dtype=bool), [True, False])

    with pytest.raises(TypeError):
        _ = metrics.dice([True, False], [True, False])


def test_dice_input_dtypes():
    """Test the dtypes accepted by the inputs to ``metrics.dice ``."""
    bool_tensor = tensor([0, 1, 0], dtype=bool)

    # Should work with boolean tensors
    _ = metrics.dice(bool_tensor, bool_tensor)

    # Should break with other dtypes
    with pytest.raises(TypeError):
        _ = metrics.dice(bool_tensor.float(), bool_tensor)

    with pytest.raises(TypeError):
        _ = metrics.dice(bool_tensor, bool_tensor.float())

    with pytest.raises(TypeError):
        _ = metrics.dice(bool_tensor.long(), bool_tensor.long())


def test_dice_arg_shapes():
    """Make sure the shapes of each input are the same."""
    # Should work with boolean tensors whose shapes match
    _ = metrics.dice(
        tensor([0, 1, 0], dtype=bool),
        tensor([0, 1, 0], dtype=bool),
    )

    # Should beak with mismatchign shapes
    with pytest.raises(RuntimeError):
        _ = metrics.dice(
            tensor([0, 1, 0, 1], dtype=bool),
            tensor([0, 1, 0], dtype=bool),
        )

    with pytest.raises(RuntimeError):
        _ = metrics.dice(
            tensor([0, 1, 0], dtype=bool),
            tensor([0, 1, 0, 1], dtype=bool),
        )


def test_dice_return_values():
    """Test the values return by ``metrics.dice``."""
    # Test with perfect agreement
    dice = metrics.dice(
        tensor([0, 1, 0], dtype=bool),
        tensor([0, 1, 0], dtype=bool),
    )
    assert dice == 1.0

    # Test with zero agreement.
    dice = metrics.dice(
        tensor([0, 1, 0], dtype=bool),
        ~tensor([0, 1, 0], dtype=bool),
    )
    assert dice == 0.0

    # Test with one point agreeing
    dice = metrics.dice(
        tensor([1, 0, 0], dtype=bool),
        tensor([1, 1, 1], dtype=bool),
    )
    assert dice == 0.5

    # Test with two points agreeing
    dice = metrics.dice(
        tensor([1, 0, 1], dtype=bool),
        tensor([1, 1, 1], dtype=bool),
    )
    assert dice == 0.8


def test_with_no_positives():
    """Test the dice function with no positives."""
    dice = metrics.dice(
        tensor([0, 0, 0], dtype=bool),
        tensor([0, 0, 0], dtype=bool),
    )
    assert isnan(dice)
