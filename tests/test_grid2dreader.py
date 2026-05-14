import numpy as np
import pytest
from netCDF4 import Dataset
from hdfkit.grid2dreader import Grid2DReader


@pytest.fixture
def nc_file(tmp_path):
    path = tmp_path / "MOD021KM_L.1000.2015001004500.H27V04.000000.nc"
    with Dataset(str(path), "w", format="NETCDF4") as fp:
        fp.createDimension("y", 10)
        fp.createDimension("x", 10)
        grp = fp.createGroup("GeometricCorrection")
        v = grp.createVariable("DataSet_1000_5", "f4", ("y", "x"))
        v[:] = np.random.rand(10, 10).astype(np.float32)
    return str(path)


def test_keys(nc_file):
    reader = Grid2DReader(nc_file, grid_size=10, do_grid_surround=False)
    assert len(reader.keys()) > 0


def test_read_full(nc_file):
    reader = Grid2DReader(nc_file, grid_size=10, do_grid_surround=False)
    dp = reader.read("/GeometricCorrection/DataSet_1000_5")
    data = dp[:]
    assert data.shape == (10, 10)


def test_read_slice(nc_file):
    reader = Grid2DReader(nc_file, grid_size=10, do_grid_surround=False)
    dp = reader.read("/GeometricCorrection/DataSet_1000_5")
    data = dp[0:5, 0:5]
    assert data.shape == (5, 5)


def test_read_negative_slice(nc_file):
    reader = Grid2DReader(nc_file, grid_size=10, do_grid_surround=False)
    dp = reader.read("/GeometricCorrection/DataSet_1000_5")
    data = dp[-2:12, -2:12]
    assert data.shape == (14, 14)
