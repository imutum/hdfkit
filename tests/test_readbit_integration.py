"""End-to-end test of TemplateReader.readbit via HDF5Reader.

Validates that bit-field extraction works through the public Reader API,
not just at the bitoffset() utility level.
"""
import numpy as np
import pytest
from netCDF4 import Dataset
from hdfkit.reader import HDF5Reader


@pytest.fixture
def qa_nc(tmp_path):
    """uint8 QA variable with known bit patterns.
    0b11010110 = 214:  bits[0,2)=2, bits[2,4)=1, bits[4,8)=13
    0b00001111 = 15:   bits[0,2)=3, bits[2,4)=3, bits[4,8)=0
    """
    path = tmp_path / "qa.nc"
    with Dataset(str(path), "w", format="NETCDF4") as fp:
        fp.createDimension("n", 4)
        v = fp.createVariable("qa", "u1", ("n",))
        v[:] = np.array([214, 15, 214, 15], dtype=np.uint8)
    return str(path)


def test_readbit_low_two_bits(qa_nc):
    bits = HDF5Reader(qa_nc).readbit("qa", 0, 2)
    np.testing.assert_array_equal(bits, [2, 3, 2, 3])


def test_readbit_middle(qa_nc):
    bits = HDF5Reader(qa_nc).readbit("qa", 2, 4)
    np.testing.assert_array_equal(bits, [1, 3, 1, 3])


def test_readbit_high_four_bits(qa_nc):
    bits = HDF5Reader(qa_nc).readbit("qa", 4, 8)
    np.testing.assert_array_equal(bits, [13, 0, 13, 0])
