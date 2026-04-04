import pytest
from pathlib import Path
from mtmhdf.grid2dreader import Grid2DReader

DATA_FILE = Path("tests/data/MOD021KM_L.1000.2015001004500.H27V04.000000.nc")

pytestmark = pytest.mark.skipif(
    not DATA_FILE.exists(),
    reason="Test data file not found"
)


def test_keys():
    reader = Grid2DReader(str(DATA_FILE), do_grid_surround=False)
    assert len(reader.keys()) > 0


def test_read_slice():
    reader = Grid2DReader(str(DATA_FILE), do_grid_surround=True)
    dp = reader.read("/GeometricCorrection/DataSet_1000_5")
    data = dp[-1:1202, -1:1202]
    assert data is not None
    assert data.shape == (1203, 1203)


def test_read_full():
    reader = Grid2DReader(str(DATA_FILE), do_grid_surround=True)
    dp = reader.read("/GeometricCorrection/DataSet_1000_5")
    data = dp[:]
    assert data is not None
