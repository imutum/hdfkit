import numpy as np
import pytest
from netCDF4 import Dataset
from hdfkit.grid2dreader import Grid2DReader


GRID = 10
CENTER_HV = "h27v04"

# Each tile filled with a distinct value so stitching is verifiable.
DIRECTION_HV = {
    "topleft":     "h26v03",
    "top":         "h27v03",
    "topright":    "h28v03",
    "left":        "h26v04",
    "center":      "h27v04",
    "right":       "h28v04",
    "bottomleft":  "h26v05",
    "bottom":      "h27v05",
    "bottomright": "h28v05",
}
TILE_VALUE = {d: i + 1 for i, d in enumerate(DIRECTION_HV)}  # 1..9


def _make_tile(path, value):
    with Dataset(str(path), "w", format="NETCDF4") as fp:
        fp.createDimension("y", GRID)
        fp.createDimension("x", GRID)
        v = fp.createVariable("data", "f4", ("y", "x"))
        v[:] = np.full((GRID, GRID), value, dtype=np.float32)


@pytest.fixture
def tile_center(tmp_path):
    """9 tiles, each filled with its direction's unique value. Returns center path."""
    for direction, hv in DIRECTION_HV.items():
        _make_tile(tmp_path / f"tile_{hv}.nc", TILE_VALUE[direction])
    return str(tmp_path / f"tile_{CENTER_HV}.nc")


def test_slice_within_center(tile_center):
    reader = Grid2DReader(tile_center, grid_size=GRID)
    arr = reader.read("data")[2:8, 2:8]
    assert arr.shape == (6, 6)
    np.testing.assert_array_equal(arr, TILE_VALUE["center"])


def test_cross_right_boundary(tile_center):
    reader = Grid2DReader(tile_center, grid_size=GRID)
    arr = reader.read("data")[2:8, 5:15]
    assert arr.shape == (6, 10)
    np.testing.assert_array_equal(arr[:, :5], TILE_VALUE["center"])
    np.testing.assert_array_equal(arr[:, 5:], TILE_VALUE["right"])


def test_cross_bottom_boundary(tile_center):
    reader = Grid2DReader(tile_center, grid_size=GRID)
    arr = reader.read("data")[5:15, 2:8]
    assert arr.shape == (10, 6)
    np.testing.assert_array_equal(arr[:5, :], TILE_VALUE["center"])
    np.testing.assert_array_equal(arr[5:, :], TILE_VALUE["bottom"])


def test_cross_bottom_right_corner(tile_center):
    """Touches center + right + bottom + bottomright."""
    reader = Grid2DReader(tile_center, grid_size=GRID)
    arr = reader.read("data")[5:15, 5:15]
    assert arr.shape == (10, 10)
    np.testing.assert_array_equal(arr[:5, :5], TILE_VALUE["center"])
    np.testing.assert_array_equal(arr[:5, 5:], TILE_VALUE["right"])
    np.testing.assert_array_equal(arr[5:, :5], TILE_VALUE["bottom"])
    np.testing.assert_array_equal(arr[5:, 5:], TILE_VALUE["bottomright"])


def test_cross_topleft_corner(tile_center):
    """Negative indices reach into top + left + topleft tiles."""
    reader = Grid2DReader(tile_center, grid_size=GRID)
    arr = reader.read("data")[-3:3, -3:3]
    assert arr.shape == (6, 6)
    np.testing.assert_array_equal(arr[:3, :3], TILE_VALUE["topleft"])
    np.testing.assert_array_equal(arr[:3, 3:], TILE_VALUE["top"])
    np.testing.assert_array_equal(arr[3:, :3], TILE_VALUE["left"])
    np.testing.assert_array_equal(arr[3:, 3:], TILE_VALUE["center"])


def test_missing_neighbor_tolerated(tmp_path):
    """When only the center tile exists, cross-boundary slicing must not crash;
    the in-bounds portion is correct and the missing region is masked."""
    center = tmp_path / f"tile_{CENTER_HV}.nc"
    _make_tile(center, TILE_VALUE["center"])

    reader = Grid2DReader(str(center), grid_size=GRID)
    arr = reader.read("data")[2:8, 5:15]

    assert arr.shape == (6, 10)
    # center portion intact
    np.testing.assert_array_equal(arr[:, :5], TILE_VALUE["center"])
    # missing right tile → those positions stay masked
    assert isinstance(arr, np.ma.MaskedArray)
    assert arr.mask[:, 5:].all()


def test_uppercase_filename_stitching(tmp_path):
    """Filenames with HXXVYY (uppercase) must resolve neighbors via the
    uppercase branch of replace_hv_surround."""
    upper_hv = {d: hv.upper() for d, hv in DIRECTION_HV.items()}
    for direction, hv in upper_hv.items():
        _make_tile(tmp_path / f"MOD.{hv}.nc", TILE_VALUE[direction])

    center_path = str(tmp_path / f"MOD.{upper_hv['center']}.nc")
    reader = Grid2DReader(center_path, grid_size=GRID)
    arr = reader.read("data")[5:15, 5:15]

    assert arr.shape == (10, 10)
    np.testing.assert_array_equal(arr[:5, :5], TILE_VALUE["center"])
    np.testing.assert_array_equal(arr[:5, 5:], TILE_VALUE["right"])
    np.testing.assert_array_equal(arr[5:, :5], TILE_VALUE["bottom"])
    np.testing.assert_array_equal(arr[5:, 5:], TILE_VALUE["bottomright"])
