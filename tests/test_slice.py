import pytest
from hdfkit._utils import split_slice_2d


GRID_SIZE = 1200


def get_slices(item):
    sub_slices, target_shape = split_slice_2d(item, GRID_SIZE)
    return sub_slices, target_shape


def test_normal_slice():
    slices, shape = get_slices((slice(0, 100), slice(0, 100)))
    assert shape == (100, 100)
    assert "center" in slices


def test_negative_start():
    slices, shape = get_slices((slice(-5, 1202), slice(-5, 1202)))
    assert shape == (1207, 1207)
    assert "topleft" in slices
    assert "center" in slices
    assert "bottomright" in slices


def test_full_slice():
    slices, shape = get_slices((slice(None), slice(None)))
    assert shape == (GRID_SIZE, GRID_SIZE)
    assert "center" in slices


def test_beyond_grid():
    slices, shape = get_slices((slice(1100, 1300), slice(0, 100)))
    assert shape == (200, 100)
    assert "center" in slices
    assert "bottom" in slices


def test_step_not_supported():
    with pytest.raises(NotImplementedError):
        get_slices((slice(0, 100, 2), slice(0, 100)))
