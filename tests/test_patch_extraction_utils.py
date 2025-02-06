"""Tests for ``ispy_msi.patch_extraction_utils``."""

import pytest

from numpy import ones

from ispy_msi.patch_extraction_utils import _patch_edges, _pad_image


def test_patch_edges_width():
    """Test the patch edge coords in the width direction."""
    # Should do nothing if the coords are inside the image.
    assert _patch_edges(0, 0, 100, 100, 50) == (0, 50, 0, 50)

    # Should shift the coords back if they are outside the image
    assert _patch_edges(0, 75, 100, 100, 50) == (50, 100, 0, 50)


def test_patch_edges_height():
    """Test the patch edge coords in the height direction."""
    # Should do nothing if the coords are inside the image.
    assert _patch_edges(0, 0, 100, 100, 50) == (0, 50, 0, 50)

    # Should shift the coords back if they are outside the image
    assert _patch_edges(99, 0, 100, 100, 50) == (0, 50, 50, 100)


def test_patch_size_values():
    """Test the values allowed by patch size."""
    # Should work if patch_size is in [2, min(width, height)]
    _patch_edges(0, 0, 128, 128, patch_size=32)

    # Should break if patch size is less than 2
    with pytest.raises(ValueError):
        _patch_edges(0, 0, 128, 128, patch_size=1)

    # Should break if patch size is bigger than the horizontal dim
    with pytest.raises(ValueError):
        _patch_edges(0, 0, 512, 64, patch_size=65)

    # Should break if patch size is bigger than the vertical dim
    with pytest.raises(ValueError):
        _patch_edges(0, 0, 64, 512, patch_size=65)


def test_pad_image_return_shapes():
    """Test the shapes returned by ``_pad_image``."""
    # Test with no pad
    img = ones((10, 10))
    new_img = _pad_image(img, patch_size=5, stride=5)

    assert img.shape == new_img.shape

    # Test with pad of (1, 2)
    img = ones((9, 8))
    new_img = _pad_image(img, patch_size=5, stride=5)

    assert new_img.shape == (10, 10)

    # Test with pad caused by stride and not patch size
    img = ones((10, 11))
    new_img = _pad_image(img, patch_size=5, stride=3)

    assert new_img.shape == (14, 14)
