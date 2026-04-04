import numpy as np
from mtmhdf._utils import int2binarystring, bitoffset


def test_int2binarystring():
    assert int2binarystring(0, 8) == "00000000"
    assert int2binarystring(255, 8) == "11111111"
    assert int2binarystring(5, 4) == "0101"
    assert int2binarystring(5) == "101"


def test_bitoffset_scalar():
    # 0b11010110 = 214
    # bits [1, 4) → bits 1,2,3 → 0b011 = 3
    assert bitoffset(214, 1, 4) == 3
    # bits [4, 8) → bits 4,5,6,7 → 0b1101 = 13
    assert bitoffset(214, 4, 8) == 13


def test_bitoffset_array():
    data = np.array([0b11110000, 0b00001111], dtype=np.uint8)
    result = bitoffset(data, 0, 4)
    np.testing.assert_array_equal(result, [0, 15])
    result = bitoffset(data, 4, 8)
    np.testing.assert_array_equal(result, [15, 0])
