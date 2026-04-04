import pytest
from pathlib import Path
from mtmhdf.grid_modis_sin import TileGridModisSin

DATA_FILE = Path("tests/data/MOD021KM_L.1000.2015001004500.H27V04.000000.nc")

pytestmark = pytest.mark.skipif(
    not DATA_FILE.exists(),
    reason="Test data file not found"
)


def test_tile_grid_modis_sin():
    t = TileGridModisSin(gcenter="h27v04", fcenter=str(DATA_FILE))
    assert t is not None
