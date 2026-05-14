import numpy as np
from netCDF4 import Dataset, Variable

from ._base import TemplateData, TemplateReader
from ._utils import mask, scale


class HDF5(Dataset):
    @staticmethod
    def open(file_path: str, mode='r', *args, **kwargs):
        return Dataset(file_path, mode=mode, *args, **kwargs)

    @staticmethod
    def read(fp: Dataset, name: str) -> Variable:
        return HDF5._jump(fp, name)

    @staticmethod
    def keys(fp: Dataset) -> list[str]:
        return list(HDF5._walk(fp))

    @staticmethod
    def dpinfo(dp: Variable) -> dict:
        info_dict = dp.__dict__
        info_dict.update({
            "dataset_name": (dp.group().path + "/" + dp.name).replace("//", "/"),
            "dataset_dims": dp.shape,
            "dataset_type": dp.datatype.name,
        })
        return info_dict

    @staticmethod
    def infos(fp: Dataset) -> dict:
        return {name: HDF5.dpinfo(HDF5.read(fp, name)) for name in HDF5.keys(fp)}

    @staticmethod
    def _walk(fp: Dataset, path=""):
        if not len(path) or path[-1] != "/":
            path += "/"
        current_variables = list(fp.variables.keys())
        for variable in current_variables:
            yield path + variable
        current_groups = list(fp.groups.keys())
        for group in current_groups:
            yield from HDF5._walk(fp.groups[group], path + group)

    @staticmethod
    def _jump(fp: Dataset, path="/"):
        path_list = path.lstrip("/").split("/")
        if not len(path_list):
            return fp
        subnode_fp = fp.__getitem__(path_list[0])
        subnode_path = "/" + "/".join(path_list[1:])
        if len(path_list) == 1:
            return subnode_fp
        else:
            return HDF5._jump(subnode_fp, subnode_path)

    @staticmethod
    def write(fp: Dataset, data: np.ndarray | np.ma.MaskedArray, varname: str, dimensions: tuple[str, ...], datatype: str = None, scale_factor=1.0, add_offset=0.0, **kwargs):
        # convert data to numpy.ma.MaskedArray
        if isinstance(data, np.ma.MaskedArray):
            xm = data
        elif isinstance(data, np.ndarray):
            xm = np.ma.masked_invalid(data)
        else:
            raise ValueError("data must be a numpy.ndarray or numpy.ma.MaskedArray")

        # check dimensions
        shape = xm.shape
        if len(dimensions) != len(shape):
            raise ValueError("dimensions (tuple) and array shape (tuple) must have the same length")
        for i, dimension in enumerate(dimensions):
            if dimension not in fp.dimensions:
                fp.createDimension(dimension, shape[i])
            else:
                if (dsize := shape[i]) != (fsize := fp.dimensions[dimension].size):
                    raise ValueError(f"dimensions ({dimension}: {dsize}) must have the same size in the file ({dimension}: {fsize})")

        # check datatype
        if datatype is None:
            datatype = xm.dtype
        else:
            # 整数目标类型按 [scale, offset] 物理范围掩膜越界值，防止 overflow
            dt = np.dtype(datatype)
            if np.issubdtype(dt, np.integer):
                info = np.iinfo(dt)
                lim1 = info.min * scale_factor + add_offset
                lim2 = info.max * scale_factor + add_offset
                xm = np.ma.masked_outside(xm, lim1, lim2)

        v = fp.createVariable(varname=varname, datatype=datatype, dimensions=dimensions, **kwargs)
        v.scale_factor = scale_factor
        v.add_offset = add_offset
        v.set_auto_maskandscale(True)

        xm.data[xm.mask] = 0  # fill invalid values with 0
        xm.fill_value = 0
        v[:] = xm
        return v


class HDF5Data(TemplateData):
    def __init__(self, dp, mode="manual", isScaleAndOffset: bool = False, isMasked: bool = True, manual_options=None, **kwargs) -> None:
        self.dp = dp
        self.mode = mode
        self.isScaleAndOffset = isScaleAndOffset
        self.isMasked = isMasked

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

        if mode == "manual":
            self.dp.set_auto_scale(False)
            self.dp.set_auto_mask(False)
        elif mode == "native":
            self.dp.set_auto_scale(isScaleAndOffset)
            self.dp.set_auto_mask(isMasked)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def infos(self):
        return HDF5.dpinfo(self.dp)

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


class HDF5Reader(TemplateReader):
    LinkedDataClass = HDF5Data

    def __init__(self, data_file: str, *args, **kwargs):
        self.fp = HDF5.open(data_file, *args, **kwargs)

    def readraw(self, name: str):
        return HDF5.read(self.fp, name)

    def read(self, name: str, isScaleAndOffset: bool = True, isMasked: bool = True, mode="native", **kwargs):
        dp = HDF5.read(self.fp, name)
        DataClass = HDF5Reader.LinkedDataClass
        return DataClass(dp, mode=mode, isScaleAndOffset=isScaleAndOffset, isMasked=isMasked, **kwargs)

    def keys(self) -> list[str]:
        return HDF5.keys(self.fp)

    def infos(self) -> dict:
        return HDF5.infos(self.fp)
