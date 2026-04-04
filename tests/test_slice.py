import numpy as np
import sys
import os

# 将 src 添加到 path 中以导入 mtmhdf
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from mtmhdf._utils import split_slice_2d


class TestSlice:
    def __init__(self, grid_size):
        self.grid_size = grid_size

    def __getitem__(self, item) -> np.ma.MaskedArray | np.ndarray:
        print(f"Original slice: {item}")
        if isinstance(item, tuple) and len(item) == 2:
            sub_slices, target_shape = split_slice_2d(item, self.grid_size)
            print(f"Target Shape: {target_shape}")
            print("Split slices:")
            for direction, info in sub_slices.items():
                print(f"  {direction:12}: src={info['src']}, dst={info['dst']}")
        return np.array([])


if __name__ == "__main__":
    grid_size = 1200
    test = TestSlice(grid_size)
    test[-5:1202, -5:1202]
    test[:]
