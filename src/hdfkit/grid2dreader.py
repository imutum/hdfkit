from pathlib import Path
import re
from hdfkit.reader import HDF4Reader, HDF5Reader, TemplateReader
from hdfkit.grid_modis_sin import TileGridModisSin
from hdfkit._utils import split_slice_2d

import numpy as np


def inferrence_format(path:str):
    suffix = Path(path).suffix.lower()
    if suffix in [".hdf", ".hdf4"]:
        return "hdf4"
    elif suffix in [".hdf5", ".h5", ".he5", ".nc"]:
        return "hdf5"
    else:
        raise ValueError(f"Unsupported format: {suffix}")


def inferrence_hv(path:str):
    filename = Path(path).name
    # 正则化表达式提取 H..V..或者 h..v..
    pattern = r'H(\d{2})V(\d{2})|h(\d{2})v(\d{2})'
    match = re.search(pattern, filename)
    result = None
    if match:
        result = match.group(0)
    if result is not None and len(result) == 6:
        return result
    else:
        raise ValueError(f"Unsupported filename: {filename}")

class Grid2DReaderData:
    def __init__(self, datas:dict, grid_size:int):
        self.datas = datas
        self.grid_size = grid_size

    def __getitem__(self, item) -> np.ma.MaskedArray | np.ndarray:
        if isinstance(item, slice):
            item = (item, slice(None))

        if not (isinstance(item, tuple) and len(item) == 2):
            center_data = self.datas.get("center")
            if center_data is None:
                center_data = list(self.datas.values())[0]
            return center_data[item]

        row_sl, col_sl = item
        
        # 1. 使用工具函数切分行列，并计算目标数组位置
        sub_slices, target_shape = split_slice_2d(item, self.grid_size)
        r_len, c_len = target_shape

        if r_len == 0 or c_len == 0:
            return np.array([]).reshape(r_len, c_len)

        # 2. 准备结果数组
        res_data = None
        
        # 3. 填充数据
        for direction, info in sub_slices.items():
            if direction in self.datas:
                data_tile = self.datas[direction]
                src_item = info['src']
                dst_item = info['dst']
                
                chunk = data_tile[src_item]
                
                if res_data is None:
                    if hasattr(chunk, "fill_value"):
                        fill_value = chunk.fill_value
                    else:
                        fill_value = 0
                    if isinstance(chunk, np.ma.MaskedArray):
                        res_data = np.ma.masked_all(target_shape, dtype=chunk.dtype)
                        res_data.fill_value = chunk.fill_value
                    else:
                        res_data = np.full(target_shape, fill_value=fill_value, dtype=chunk.dtype)
                
                res_data[dst_item] = chunk
        
        return res_data if res_data is not None else np.array([]).reshape(target_shape)

class Grid2DReader:
    def __init__(self, path:str, grid_format:str=None, grid_size:int=1200, do_grid_surround:bool=True, **kwargs):
        self.path = path
        self.grid_format = grid_format if grid_format is not None else "modis_sin"
        self.file_format = inferrence_format(path)
        self.hv = inferrence_hv(path)
        match self.file_format:
            case "hdf4":
                self.reader_class: TemplateReader = HDF4Reader
            case "hdf5":
                self.reader_class: TemplateReader = HDF5Reader
            case _:
                raise ValueError(f"Unsupported format: {self.file_format}")
        
        match self.grid_format:
            case "modis_sin" | "MODIS_SIN":
                self.tile_grid = TileGridModisSin(
                    gcenter=self.hv, 
                    fcenter=path, 
                    gsize=grid_size, 
                    do_grid_surround=do_grid_surround
                )
            case _:
                raise ValueError(f"Unsupported grid format: {self.grid_format}")
        
        self.__init_grid_file_reader(**kwargs)

    def __init_grid_file_reader(self, **kwargs):
        if self.tile_grid.fcenter is not None:
            self.readerc = self.reader_class(self.tile_grid.fcenter, **kwargs)
        if self.tile_grid.fleft is not None:
            self.readerl = self.reader_class(self.tile_grid.fleft, **kwargs)
        if self.tile_grid.fright is not None:
            self.readerr = self.reader_class(self.tile_grid.fright, **kwargs)
        if self.tile_grid.ftop is not None:
            self.readert = self.reader_class(self.tile_grid.ftop, **kwargs)
        if self.tile_grid.fbottom is not None:
            self.readerb = self.reader_class(self.tile_grid.fbottom, **kwargs)
        if self.tile_grid.ftopleft is not None:
            self.readertl = self.reader_class(self.tile_grid.ftopleft, **kwargs)
        if self.tile_grid.ftopright is not None:
            self.readertr = self.reader_class(self.tile_grid.ftopright, **kwargs)
        if self.tile_grid.fbottomleft is not None:
            self.readerbl = self.reader_class(self.tile_grid.fbottomleft, **kwargs)
        if self.tile_grid.fbottomright is not None:
            self.readerbr = self.reader_class(self.tile_grid.fbottomright, **kwargs)

    def read(self, name:str, **kwargs):
        datas = {}
        mapping = {
            "readerc": "center",
            "readerl": "left",
            "readerr": "right",
            "readert": "top",
            "readerb": "bottom",
            "readertl": "topleft",
            "readertr": "topright",
            "readerbl": "bottomleft",
            "readerbr": "bottomright",
        }
        for attr, direction in mapping.items():
            if hasattr(self, attr):
                reader = getattr(self, attr)
                datas[direction] = reader.read(name, **kwargs)
        
        if not datas:
            raise ValueError(f"No readers initialized for {self.path}")
            
        return Grid2DReaderData(datas, self.tile_grid.gsize)

    def keys(self) -> list[str]:
        if hasattr(self, "readerc"):
            return self.readerc.keys()
        return []

    def infos(self) -> dict:
        if hasattr(self, "readerc"):
            return self.readerc.infos()
        return {}
