from netCDF4 import Dataset, Variable
import numpy as np


class HDF5(Dataset):
    @staticmethod
    def open(file_path:str, mode='r', *args, **kwargs):
        return Dataset(file_path, mode=mode, *args, **kwargs)

    @staticmethod
    def read(fp:Dataset, name:str) -> Variable:
        return HDF5._jump(fp, name)

    @staticmethod
    def keys(fp:Dataset) -> list[str]:
        return list(HDF5._walk(fp))
    
    @staticmethod
    def dpinfo(dp: Variable) -> dict:
        info_dict = dp.__dict__
        info_dict.update({
            "dataset_name": (dp.group().path + "/" + dp.name).replace("//", "/"),
            "dataset_dims": dp.shape,
            "dataset_type": dp.datatype.name
        })
        return info_dict

    @staticmethod
    def infos(fp:Dataset) -> dict:
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
    def write(fp: Dataset, data:np.ndarray|np.ma.MaskedArray, varname:str, dimensions:tuple[str, ...], datatype:str = None, scale_factor=1.0, add_offset=0.0, **kwargs):
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
                if (dsize := shape[i]) != (fsize:=fp.dimensions[dimension].size):
                    raise ValueError(f"dimensions ({dimension}: {dsize}) must have the same size in the file ({dimension}: {fsize})")

        # check datatype
        if datatype is None:
            datatype = xm.dtype
        else:
            # 将 datatype (如 'i2', 'int16') 转换为 numpy dtype 对象
            dt = np.dtype(datatype)
            
            # 只有目标类型是整数时，才需要考虑截断（防止 Overflow）
            if np.issubdtype(dt, np.integer):
                # 1. 获取目标整数类型的理论最大/最小值
                info = np.iinfo(dt)
                i_min, i_max = info.min, info.max
                # 3. 转换为物理值的范围
                #    公式: Physical = Integer * scale + offset
                lim1 = i_min * scale_factor + add_offset
                lim2 = i_max * scale_factor + add_offset
                xm = np.ma.masked_outside(xm, lim1, lim2)

        # create variable
        v = fp.createVariable(varname=varname, datatype=datatype, dimensions=dimensions, **kwargs)

        # set scale_factor and add_offset
        v.scale_factor = scale_factor
        v.add_offset = add_offset
        v.set_auto_maskandscale(True)

        # write data
        xm.data[xm.mask] = 0 # fill invalid values with 0
        xm.fill_value = 0
        v[:] = xm

        return v