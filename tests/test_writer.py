import numpy as np
import pytest
from netCDF4 import Dataset
from mtmhdf.reader import HDF5Reader
from mtmhdf._hdf5 import HDF5


@pytest.fixture
def nc_path(tmp_path):
    return str(tmp_path / "test.nc")


def test_write_and_read_float(nc_path):
    x = np.random.rand(100, 100).astype(np.float32)
    with Dataset(nc_path, "w", format="NETCDF4") as fp:
        HDF5.write(fp, x, "x", ("t", "p"))
    y = HDF5Reader(nc_path).read("x")[:]
    assert y.shape == (100, 100)
    assert np.allclose(x, y, atol=1e-5)


def test_write_with_scale_factor(nc_path):
    x = np.random.rand(100, 100).astype(np.float32)
    x[1, :] = np.nan
    with Dataset(nc_path, "w", format="NETCDF4") as fp:
        HDF5.write(fp, x, "x", ("t", "p"), "uint16", scale_factor=0.0001)
    y = HDF5Reader(nc_path).read("x")[:]
    assert y.shape == (100, 100)
    assert isinstance(y, np.ma.MaskedArray)
    assert y.mask[1, :].all()


def test_write_dimension_mismatch(nc_path):
    x = np.random.rand(100, 100).astype(np.float32)
    with Dataset(nc_path, "w", format="NETCDF4") as fp:
        with pytest.raises(ValueError, match="dimensions"):
            HDF5.write(fp, x, "x", ("t",))
