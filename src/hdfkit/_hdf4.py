import numpy as np
from pyhdf.SD import SD, SDC, SDS

from ._base import TemplateData, TemplateReader
from ._utils import mask, scale


class HDF4:
    DATATYPES = {
        4: "char",
        3: "uchar",
        20: "int8",
        21: "uint8",
        22: "int16",
        23: "uint16",
        24: "int32",
        25: "uint32",
        5: "float32",
        6: "float64",
    }
    OPENMODES = {"r": SDC.READ, "w": SDC.WRITE}

    @staticmethod
    def open(file_path: str, mode='r', *args, **kwargs):
        return SD(file_path, mode=HDF4.OPENMODES[mode], *args, **kwargs)

    @staticmethod
    def read(fp: SD, dataset_name: str):
        return fp.select(dataset_name)

    @staticmethod
    def keys(fp: SD) -> list[str]:
        return list(fp.datasets().keys())

    @staticmethod
    def infos(fp: SD) -> dict:
        return {name: HDF4.dpinfo(HDF4.read(fp, name)) for name in HDF4.keys(fp)}

    @staticmethod
    def dpinfo(dp: SDS) -> dict:
        attrs = dp.attributes()
        _info_list = dp.info()
        _info_dict = {
            "dataset_name": _info_list[0],
            "dataset_rank": _info_list[1],
            "dataset_dims": _info_list[2],
            "dataset_type": HDF4.DATATYPES[_info_list[3]],
        }
        _info_dict.update(attrs)
        return _info_dict


class HDF4Data(TemplateData):
    def __init__(self, dp: HDF4, mode="manual", isScaleAndOffset: bool = True, isMasked: bool = True, manual_options=None, **kwargs) -> None:
        self.dp = dp
        self.mode = mode
        self.isMasked = isMasked
        self.isScaleAndOffset = isScaleAndOffset

        default_manual_options = {
            "attr_scale_factor": "scale_factor",
            "attr_add_offset": "add_offset",
            "attr_fill_value": "_FillValue",
            "attr_decimal": 8,
        }
        if manual_options is None:
            self.manual_options = default_manual_options
        else:
            self.manual_options = manual_options.copy()
            self.manual_options.update(default_manual_options)

    def infos(self):
        return HDF4.dpinfo(self.dp)

    def manual_transform(self, data: np.ndarray) -> np.ma.MaskedArray | np.ndarray:
        infos: dict = self.infos()

        attr_scale_factor = self.manual_options["attr_scale_factor"]
        attr_add_offset = self.manual_options["attr_add_offset"]
        attr_fill_value = self.manual_options["attr_fill_value"]
        attr_decimal = self.manual_options["attr_decimal"]

        scale_factor = round(infos.get(attr_scale_factor, 1), attr_decimal)
        add_offset = round(infos.get(attr_add_offset, 0), attr_decimal)
        fill_value = infos.get(attr_fill_value)

        if self.isMasked:
            data = mask(data, fill_value)
        if self.isScaleAndOffset:
            data = scale(data, scale_factor, add_offset)
        return data

    def __getitem__(self, *item) -> np.ma.MaskedArray | np.ndarray:
        data = self.dp.__getitem__(*item)
        if self.mode == "native":
            return data
        elif self.mode == "manual":
            return self.manual_transform(np.array(data))
        else:
            raise ValueError(f"Invalid mode: {self.mode}")


class HDF4Reader(TemplateReader):
    LinkedDataClass = HDF4Data

    def __init__(self, data_file: str, *args, **kwargs):
        self.fp = HDF4.open(data_file, *args, **kwargs)

    def readraw(self, name: str):
        return HDF4.read(self.fp, name)

    def read(self, name: str, isScaleAndOffset: bool = True, isMasked: bool = True, **kwargs):
        dp = HDF4.read(self.fp, name)
        DataClass = HDF4Reader.LinkedDataClass
        return DataClass(dp, mode="manual", isScaleAndOffset=isScaleAndOffset, isMasked=isMasked, **kwargs)

    def keys(self) -> list[str]:
        return HDF4.keys(self.fp)

    def infos(self) -> dict:
        return HDF4.infos(self.fp)
