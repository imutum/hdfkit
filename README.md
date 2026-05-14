# hdfkit

> 面向遥感与科学数据的统一 HDF4 / HDF5 读取库，附带 MODIS 正弦投影瓦片的自动跨界拼接能力。

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

## 核心特性

- **统一接口** — `HDF4Reader` 与 `HDF5Reader` 同名同形，切换格式只换一个类名。
- **自动标定** — 读取时自动应用 `scale_factor` / `add_offset` 并按 `_FillValue` 掩膜，返回可直接计算的 `MaskedArray`。
- **瓦片拼接** — `Grid2DReader` 识别 MODIS `hXXvYY` 文件名，跨边界切片时自动从相邻瓦片取数并拼接。
- **Bit 字段解码** — `readbit()` 一行提取 QA 标志位，无需手写位运算。
- **安全写入** — `HDF5.write` 在写整型时自动按物理范围掩膜，杜绝溢出脏数据。

## 安装

依赖：Python **3.11+**、`pyhdf >= 0.10.5`、`netcdf4 >= 1.6.5`、`numpy >= 1.23`

```bash
pip install hdfkit
# 或
uv add hdfkit
```

源码安装：

```bash
git clone <repo>
cd hdfkit
uv sync
```

## 快速上手

### 读取 HDF4 / HDF5

```python
from hdfkit import HDF4Reader, HDF5Reader

# HDF4
h4 = HDF4Reader("MOD09GA.A2023001.h25v03.006.hdf")
h4.keys()                  # 列出所有数据集名
h4.infos()                 # 一次性返回所有数据集的元信息

ndvi = h4.read("sur_refl_b01")   # 返回 HDF4Data（懒加载）
arr = ndvi[:]                    # 实际读盘 → MaskedArray，已应用 scale/offset/mask
arr = ndvi[100:200, 300:400]     # 也支持切片

# HDF5 / NetCDF — 接口完全一致
h5 = HDF5Reader("data.nc")
temp = h5.read("temperature")[:]
```

### 关闭自动变换

```python
raw = h4.read("sur_refl_b01", isScaleAndOffset=False, isMasked=False)[:]  # 原始整数 DN
sds = h4.readraw("sur_refl_b01")                                          # 直接返回 pyhdf SDS 对象
```

### 解码 QA Bit 字段

```python
# MODIS State_1km bit 0-1 是 Cloud State：0=Clear, 1=Cloudy, 2=Mixed, 3=Unknown
cloud_state = h4.readbit("State_1km", bit_start_pos=0, bit_end_pos=2)
```

> 区间语义：左闭右开，最低有效位为 bit 0。

### MODIS 瓦片跨界拼接

`Grid2DReader` 接收一个中心瓦片路径，自动定位同目录下的 8 个相邻瓦片。切片越界时透明拼接，缺失瓦片自动跳过。

```python
from hdfkit import Grid2DReader

reader = Grid2DReader("MOD09GA.A2023001.h25v03.006.hdf", grid_size=1200)

ndvi = reader.read("sur_refl_b01")

# 中心瓦片内部切片
ndvi[200:800, 200:800].shape          # (600, 600)

# 跨右下边界 → 自动从 h26v03、h25v04、h26v04 拼接
ndvi[1000:1500, 1000:1500].shape      # (500, 500)

# 跨左边界 → 自动从 h24v03 拼接
ndvi[:, -100:100].shape               # (1200, 200)
```

支持经度环绕（`h35` ↔ `h00`）。

### 写入 HDF5

```python
from netCDF4 import Dataset
from hdfkit._hdf5 import HDF5
import numpy as np

fp = Dataset("out.nc", "w")
data = np.random.rand(1200, 1200).astype(np.float32)

HDF5.write(
    fp, data,
    varname="ndvi",
    dimensions=("y", "x"),
    datatype="i2",            # int16，自动按 [scale, offset] 范围掩膜超界值
    scale_factor=0.0001,
    add_offset=0.0,
)
fp.close()
```

## API 速查

| 类 / 方法 | 用途 |
|-----------|------|
| `HDF4Reader(path)` / `HDF5Reader(path)` | 打开文件，统一接口 |
| `.keys()` | 列出全部数据集名 |
| `.infos()` | 所有数据集的元信息字典 |
| `.read(name, isScaleAndOffset=True, isMasked=True)` | 返回懒加载 `*Data` 对象 |
| `.readraw(name)` | 返回底层 pyhdf SDS / netCDF4 Variable |
| `.readbit(name, lo, hi)` | 提取 bit 字段（左闭右开） |
| `Grid2DReader(path, grid_size=1200)` | 瓦片自动拼接读取器 |
| `HDF5.write(fp, data, ...)` | 安全写入 NetCDF/HDF5（自动范围保护） |

## 支持的扩展名

| 类型 | 后缀 | 处理类 |
|------|------|--------|
| HDF4 | `.hdf` `.hdf4` | `HDF4Reader` |
| HDF5 / NetCDF | `.h5` `.he5` `.hdf5` `.nc` | `HDF5Reader` |

`Grid2DReader` 内部自动从后缀推断格式。

## 架构

```
Grid2DReader            ── 瓦片九宫格拼接（高层）
    │
    ├── HDF4Reader      ── 统一标定接口（中层）
    └── HDF5Reader
            │
            ├── HDF4    ── pyhdf 薄封装（底层）
            └── HDF5    ── netCDF4 薄封装
```

`TemplateReader` / `TemplateData` 为抽象基类，可继承实现自定义格式（如 Zarr、GeoTIFF）。

## 限制

- 瓦片网格当前仅支持 MODIS 正弦投影（`modis_sin`）。
- HDF4 仅读取，不支持写入。
- 无内置缓存，重复切片会重复读盘。

## License

MIT © imutum
