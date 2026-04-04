import pytest
from mtmhdf.grid_modis_sin import TileGridModisSin, get_grid_hv_surround


def test_surround_basic():
    assert get_grid_hv_surround("h27v04", "left") == "h26v04"
    assert get_grid_hv_surround("h27v04", "right") == "h28v04"
    assert get_grid_hv_surround("h27v04", "top") == "h27v03"
    assert get_grid_hv_surround("h27v04", "bottom") == "h27v05"


def test_surround_wrap_horizontal():
    assert get_grid_hv_surround("h35v04", "right") == "h00v04"
    assert get_grid_hv_surround("h00v04", "left") == "h35v04"


def test_surround_wrap_vertical():
    assert get_grid_hv_surround("h27v17", "bottom") == "h27v00"
    assert get_grid_hv_surround("h27v00", "top") == "h27v17"


def test_tile_grid_no_file():
    t = TileGridModisSin(gcenter="h27v04", do_grid_surround=False)
    assert t.gcenter == "h27v04"
    assert t.gsize == 1200


def test_tile_grid_with_surround_no_file():
    t = TileGridModisSin(gcenter="h27v04", do_grid_surround=True)
    assert t.gleft == "h26v04"
    assert t.gright == "h28v04"
    assert t.gtopleft == "h26v03"
    # no fcenter → surrounding file paths are all None
    assert t.fleft is None


def test_tile_grid_invalid_gcenter():
    with pytest.raises(ValueError, match="gcenter must be a string"):
        TileGridModisSin(gcenter=123, do_grid_surround=True)
