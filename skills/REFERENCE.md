# hdfkit API Reference

## Installation

```bash
uv pip install hdfkit --index-url https://pypi.mutum.top:16181/simple/

# Or in pyproject.toml
# [[tool.uv.index]]
# name = "mutum-pypi"
# url = "https://pypi.mutum.top:16181/simple/"
```

## HDF4Reader

```python
from hdfkit import HDF4Reader

reader = HDF4Reader("MOD09GA.A2023001.h25v03.006.hdf")

reader.keys()        # list[str] — all dataset names
reader.infos()       # dict — metadata for all datasets

# Read with auto scale/offset/mask
data = reader.read("sur_refl_b01")
arr = data[:]                    # full array → np.ma.MaskedArray
arr = data[100:200, 300:400]     # 2D slice

# Read without transformations
data = reader.read("sur_refl_b01", isScaleAndOffset=False, isMasked=False)
raw_arr = data[:]                # raw integers

# Raw pyhdf SDS object
sds = reader.readraw("sur_refl_b01")

# Bit-field extraction [bit_start, bit_end) — MSB=7, LSB=0
cloud_state = reader.readbit("State_1km", 0, 2)  # bits 0-1
```

## HDF5Reader

```python
from hdfkit import HDF5Reader

reader = HDF5Reader("data.h5")  # or .nc, .he5, .hdf5

reader.keys()        # list[str] — full paths like "/group/dataset"
reader.infos()       # dict

# Read (same interface as HDF4Reader)
data = reader.read("/AOD1000M/DataSet_1000_1")
arr = data[:]

# Raw netCDF4 Variable
var = reader.readraw("/AOD1000M/DataSet_1000_1")
```

## HDF4Data / HDF5Data

Lazy-loaded data wrapper, returned by `reader.read()`.

```python
data = reader.read("name", isScaleAndOffset=True, isMasked=True)

data.infos()         # dict — dataset metadata (dims, type, attributes, FillValue)
data[:]              # load full array with transformations
data[100:200, :]     # slice

# Custom attribute names (for non-standard files)
data = reader.read("name", manual_options={
    "attr_scale_factor": "ScaleFactor",
    "attr_add_offset": "Offset",
    "attr_fill_value": "FillValue",
})
```

## Custom Data Class (LinkedDataClass)

`Reader.read()` 内部通过 `self.LinkedDataClass` 实例化 Data 对象。子类可覆写 `LinkedDataClass` 指向自定义 Data 类，覆写 `manual_transform()` 实现非标准标定：

```python
from hdfkit import HDF4Data, HDF4Reader
from hdfkit._utils import mask
import numpy as np

class ReflectanceData(HDF4Data):
    """标定公式: scale * (data - offset)，非默认 data * scale + offset"""
    def manual_transform(self, data):
        infos = self.infos()
        s = np.asarray(infos.get("reflectance_scales", 1))
        o = np.asarray(infos.get("reflectance_offsets", 0))
        if self.isMasked:
            data = mask(data, infos.get("_FillValue"))
        if self.isScaleAndOffset:
            data = s * (data - o)
        return data

class ReflectanceReader(HDF4Reader):
    LinkedDataClass = ReflectanceData

reader = ReflectanceReader("MOD021KM.A2023001.0500.061.hdf")
refl = reader.read("EV_1KM_RefSB")[:]  # 使用自定义标定
```

## Grid2DReader — MODIS Tile Stitching

Reads MODIS sinusoidal tiles with automatic neighboring-tile stitching for cross-boundary slices.

```python
from hdfkit import Grid2DReader

reader = Grid2DReader(
    "MOD09GA.A2023001.h25v03.006.hdf",
    grid_size=1200,          # tile dimension (default 1200)
    do_grid_surround=True,   # auto-load 8 neighbor tiles
)

ndvi = reader.read("sur_refl_b01")

# Within-tile slice — normal
within = ndvi[200:800, 200:800]

# Cross-boundary slice — auto-stitches from h26v03, h25v04, h26v04
cross = ndvi[1000:1500, 1000:1500]

# Supports HDF4 and HDF5 (inferred from file extension)
reader = Grid2DReader("FY3D.h25v04.h5", grid_size=1200)
```

Requirements:
- Filename must contain `h##v##` or `H##V##` pattern
- Neighbor files must be in the same directory with same naming pattern

## Utility Functions

```python
from hdfkit._utils import bitoffset, scale, mask

# Bit extraction: [bit_start, bit_end) half-open interval
cloud = bitoffset(qa_array, 0, 2)      # bits 0-1 → values 0-3
shadow = bitoffset(qa_array, 2, 3)     # bit 2 → values 0-1

# Linear transform: data * scale_factor + add_offset
calibrated = scale(raw, scale_factor=0.0001, add_offset=0.0)

# Mask fill values → np.ma.MaskedArray
masked = mask(data, fill_value=65535)
```

## Writing HDF5

```python
from netCDF4 import Dataset
from hdfkit._hdf5 import HDF5
import numpy as np

fp = Dataset("output.nc", "w")
data = np.random.rand(1200, 1200).astype(np.float32)

HDF5.write(
    fp, data,
    varname="ndvi",
    dimensions=("y", "x"),    # auto-creates dimensions
    datatype="i2",            # int16 storage
    scale_factor=0.0001,
    add_offset=0.0,
)
fp.close()
```

## Common Patterns

### Read FY3D AOD product

```python
from hdfkit import HDF5Reader
import numpy as np

reader = HDF5Reader("FY3DMERSI.AOD.1000.2025001091500.H24V05.000000.h5")
fill = int(reader.infos()['/AOD1000M/DataSet_1000_1']['FillValue'])
raw = np.ma.filled(reader.readraw('/AOD1000M/DataSet_1000_1')[:], fill)
aod = raw.astype(np.float32)
aod[raw == fill] = np.nan
aod /= 10000.0
```

### Read MODIS MCD19A2 with bit QA

```python
from hdfkit import HDF4Reader

reader = HDF4Reader("MCD19A2.A2021183.h25v04.061.hdf")
aod = reader.read("Optical_Depth_047")[:]    # shape: (n_orbits, 1200, 1200)
qa = reader.readbit("AOD_QA", 5, 8)         # bits 5-7: QA confidence
```

### Cross-tile NDVI extraction

```python
from hdfkit import Grid2DReader

reader = Grid2DReader("MOD09GA.A2023001.h25v03.006.hdf")
b1 = reader.read("sur_refl_b01")[900:1300, 900:1300]  # crosses into h26v04
b2 = reader.read("sur_refl_b02")[900:1300, 900:1300]
ndvi = (b2 - b1) / (b2 + b1)
```

## Dependencies

- `pyhdf >= 0.10.5` (HDF4 support)
- `netcdf4 >= 1.6.5` (HDF5/NetCDF support)
- `numpy >= 1.23.0`

## Key Design Notes

- `read()` returns a lazy object — data only loads on `[:]` / slice
- HDF4Reader and HDF5Reader have identical APIs, only class name differs
- `readraw()` returns the underlying library object (pyhdf SDS / netCDF4 Variable) for direct access
- Bit numbering: MSB = bit 7, LSB = bit 0 within each byte
- Grid2DReader auto-detects format from extension (.hdf → HDF4, .h5/.nc → HDF5)
