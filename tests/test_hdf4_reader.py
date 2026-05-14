import numpy as np
import pytest

SD_module = pytest.importorskip(
    "pyhdf.SD",
    reason="pyhdf native DLLs unavailable",
    exc_type=ImportError,
)
SD = SD_module.SD
SDC = SD_module.SDC

from hdfkit.reader import HDF4Reader


SCALE = 0.0001
OFFSET = 0.5
FILL = -9999


@pytest.fixture
def hdf4_file(tmp_path):
    """Build a minimal HDF4 with two datasets:
    - ndvi: int16 with scale/offset/_FillValue
    - qa:   uint8 for bit-field testing
    """
    path = str(tmp_path / "test.hdf")
    raw_ndvi = np.array(
        [[1000, 2000, FILL],
         [FILL, 3000, 4000]],
        dtype=np.int16,
    )
    # 0b11010110 = 214; bits [0,2)=2, bits [4,8)=13
    raw_qa = np.array([[214, 214, 0], [0, 214, 214]], dtype=np.uint8)

    sd = SD(path, SDC.WRITE | SDC.CREATE)
    ds = sd.create("ndvi", SDC.INT16, raw_ndvi.shape)
    ds[:] = raw_ndvi
    ds.scale_factor = SCALE
    ds.add_offset = OFFSET
    ds.setfillvalue(FILL)
    ds.endaccess()

    ds = sd.create("qa", SDC.UINT8, raw_qa.shape)
    ds[:] = raw_qa
    ds.endaccess()

    sd.end()
    return path, raw_ndvi, raw_qa


def test_keys(hdf4_file):
    path, _, _ = hdf4_file
    keys = HDF4Reader(path).keys()
    assert set(keys) == {"ndvi", "qa"}


def test_infos(hdf4_file):
    path, _, _ = hdf4_file
    infos = HDF4Reader(path).infos()
    assert infos["ndvi"]["scale_factor"] == pytest.approx(SCALE)
    assert infos["ndvi"]["add_offset"] == pytest.approx(OFFSET)
    assert infos["ndvi"]["_FillValue"] == FILL
    assert infos["ndvi"]["dataset_type"] == "int16"
    assert infos["ndvi"]["dataset_dims"] == [2, 3]


def test_read_applies_scale_and_mask(hdf4_file):
    path, raw, _ = hdf4_file
    arr = HDF4Reader(path).read("ndvi")[:]

    assert isinstance(arr, np.ma.MaskedArray)
    # fill value cells are masked
    expected_mask = (raw == FILL)
    np.testing.assert_array_equal(arr.mask, expected_mask)
    # valid cells got scale * raw + offset
    valid = ~expected_mask
    expected = raw[valid] * SCALE + OFFSET
    np.testing.assert_allclose(arr.data[valid], expected, atol=1e-7)


def test_read_raw_disables_transforms(hdf4_file):
    path, raw, _ = hdf4_file
    arr = HDF4Reader(path).read("ndvi", isScaleAndOffset=False, isMasked=False)[:]
    np.testing.assert_array_equal(arr, raw)


def test_read_slice(hdf4_file):
    path, raw, _ = hdf4_file
    sub = HDF4Reader(path).read("ndvi")[0:1, :]
    assert sub.shape == (1, 3)


def test_readraw_returns_sds(hdf4_file):
    SDS = SD_module.SDS
    path, _, _ = hdf4_file
    sds = HDF4Reader(path).readraw("ndvi")
    assert isinstance(sds, SDS)


def test_readbit(hdf4_file):
    path, _, raw_qa = hdf4_file
    # bits [0,2) of 214 (0b11010110) → 0b10 = 2; cells with value 0 → 0
    bits = HDF4Reader(path).readbit("qa", 0, 2)
    expected = np.where(raw_qa == 214, 2, 0)
    np.testing.assert_array_equal(bits, expected)
