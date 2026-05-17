---
name: hdfkit-usage
description: Write code using the hdfkit library for reading/writing HDF4, HDF5, and NetCDF files with automatic scale/offset/masking, bit-field extraction, and MODIS sinusoidal tile stitching. Use when user works with HDF satellite products, .hdf/.h5/.nc files, MODIS/VIIRS/FY3D tile data, needs cross-boundary slicing, or QA bit decoding.
---

# hdfkit

HDF4/5 统一读写库，支持 MODIS 正弦投影 tile 自动拼接。源码位于 `D:\project\hdfkit`。

## Quick start

```python
from hdfkit import HDF4Reader, HDF5Reader, Grid2DReader

# HDF4 / HDF5 接口完全一致
reader = HDF4Reader("MOD09GA.A2023001.h25v03.006.hdf")
reader = HDF5Reader("FY3DMERSI.AOD.1000.2025001.H24V05.h5")

data = reader.read("dataset_name")[:]      # 自动 scale/offset/mask
raw = reader.readraw("dataset_name")[:]    # 原始值
bits = reader.readbit("QA_field", 0, 2)   # 提取 bit 0-1
```

## Key classes

| Class | Purpose |
|-------|---------|
| `HDF4Reader` | HDF4 (.hdf) 读取 |
| `HDF5Reader` | HDF5/NetCDF (.h5, .nc) 读取 |
| `Grid2DReader` | MODIS tile 跨边界自动拼接 |
| `HDF4Data` / `HDF5Data` | 延迟加载数据对象 |

## Installation

```bash
uv pip install hdfkit --index-url https://pypi.mutum.top:16181/simple/
```

## Detailed API

See [REFERENCE.md](REFERENCE.md)
