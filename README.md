# mtmhdf

`mtmhdf` 是一个 Python 二次封装 库，用于高效处理 HDF4 (pyhdf) 和 HDF5 (netCDF4) 格式的科学数据。它不仅提供了统一的读取接口，还针对 网格数据 (MODIS 正弦投影 (Sinusoidal) 等) 提供了**无缝跨图幅读取**的高级功能。

## 核心特性

*   **格式全面**：原生支持 HDF4 (.hdf) 和 HDF5/NetCDF (.h5, .nc) 格式。
*   **智能读取**：自动应用 `scale_factor` (比例因子)、`add_offset` (偏移量) 和 `_FillValue` (填充值)，直接获取物理意义的数值。
*   **无缝拼接**：**独家功能**，`Grid2DReader` 支持自动识别相邻图幅文件。当读取范围超出当前文件边界时，自动从相邻图幅（上、下、左、右及对角）读取数据并拼接，无需手动处理图幅边缘。
*   **灵活控制**：支持 `native` (原生) 和 `manual` (手动) 模式，可完全控制数据的转换过程。
*   **位操作工具**：内置高效的位操作方法，轻松提取质量控制 (QA) 数据的特定位段。

## 安装

本项目依赖 Python 3.11+。

### 从 PyPI 安装

```bash
pip install mtmhdf
# 或使用 uv
uv add mtmhdf
```

### 从源码安装

```bash
pip install .
# 或使用 uv
uv sync
```

## 快速开始

### 1. 基础文件读取

使用 `HDF4Reader` 或 `HDF5Reader` 读取单个文件。

```python
from mtmhdf import HDF4Reader, HDF5Reader

# 读取 HDF4 文件
h4 = HDF4Reader("MOD09GA.A2023001.h25v03.006.hdf")
print(h4.keys())  # 列出所有数据集

# 读取数据 (自动处理 Scale/Offset/Mask)
# 返回的是 MaskedArray，已转换为物理值
sur_refl_b01 = h4["sur_refl_b01"] 
print(sur_refl_b01.mean())

# 读取 HDF5/NetCDF 文件
h5 = HDF5Reader("data.nc")
data = h5["temperature"]
```

### 2. 无缝跨图幅读取 (Grid2DReader)

处理分幅数据（如 MODIS, VIIRS）时，经常遇到感兴趣区域 (ROI) 跨越多个图幅的情况。`Grid2DReader` 能自动检测同一目录下的相邻图幅，像操作单张大图一样操作多张图幅。

```python
from mtmhdf import Grid2DReader

# 只需要指定中心图幅的文件路径
# 程序会自动寻找同目录下的 h25v04, h26v03 等相邻文件（基于文件名中的 HxxVxx 标识）
reader = Grid2DReader("MOD09GA.A2023001.h25v03.006.hdf")

# 读取一个数据集对象 (延迟加载)
ds = reader.read("sur_refl_b01")

# 进行切片操作
# 假设单景大小为 1200*1200

# 这个切片跨越了中心图幅的右边界和下边界
# mtmhdf 会自动从右侧、下方和右下方的文件中读取对应数据并拼接
subset = ds[1000:1500, 1000:1500] 

print(subset.shape) 
# 输出: (500, 500)

# 这个切片会获取中心图幅的全部数据，不会从周边数据拼接
subset = ds[:] 

print(subset.shape) 
# 输出: (1200, 1200)
```

## 高级用法

### 手动控制数据转换

如果需要原始的整数数据（DN值），可以关闭自动转换：

```python
# 读取原始数据 (不应用 scale/offset/mask)
raw_data = h4.read("sur_refl_b01", isScaleAndOffset=False, isMasked=False)

# 或者获取原始读取对象
raw_obj = h4.readraw("sur_refl_b01")
```

### 位数据提取 (Bit Manipulation)

提取质量控制字段（如 Quality Flag）中的特定位：

```python
# 读取 "State_1km" 数据集的第 0-1 位 (Cloud State)
# readbit(name, start_bit, end_bit) 左闭右开
# 0: Clear, 1: Cloudy, 2: Mixed, ...
cloud_state = h4.readbit("State_1km", 0, 2)
```

## 支持的文件格式

*   **HDF4**: `.hdf`, `.hdf4` (自动使用 `HDF4Reader`)
*   **HDF5 / NetCDF**: `.h5`, `.he5`, `.hdf5`, `.nc` (自动使用 `HDF5Reader`)

## 开源协议

MIT License
