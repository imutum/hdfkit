import numpy as np

from ._utils import bitoffset


class TemplateData:
    def __init__(self) -> None:
        raise NotImplementedError("Subclass must implement this method")

    def infos(self) -> dict:
        raise NotImplementedError("Subclass must implement this method")

    def __getitem__(self, *item) -> np.ma.MaskedArray | np.ndarray:
        raise NotImplementedError("Subclass must implement this method")


class TemplateReader:
    LinkedDataClass = None

    def __init__(self, data_file: str, *args, **kwargs):
        raise NotImplementedError("Subclass must implement this method")

    def read(self, name: str, *args, **kwargs):
        raise NotImplementedError("Subclass must implement this method")

    def __getitem__(self, name: str):
        return self.readraw(name)

    def readraw(self, name: str):
        raise NotImplementedError("Subclass must implement this method")

    def readbit(self, name: str, bit_start_pos: int, bit_end_pos: int, *args, **kwargs) -> np.ma.MaskedArray | np.ndarray:
        # Bit fields within each byte are numbered from the left:
        # 7, 6, 5, 4, 3, 2, 1, 0.
        # The left-most bit (bit 7) is the most significant bit.
        # The right-most bit (bit 0) is the least significant bit.
        # 左闭右开区间，即从bit_start_pos开始，到bit_end_pos结束，不包含bit_end_pos
        dp = self.readraw(name)
        return bitoffset(np.array(dp[:]), bit_start_pos, bit_end_pos)

    def keys(self) -> list[str]:
        raise NotImplementedError("Subclass must implement this method")

    def infos(self) -> dict:
        raise NotImplementedError("Subclass must implement this method")
