from netCDF4 import Dataset
import numpy as np
from mtmhdf.reader import HDF5Reader
from mtmhdf._hdf5 import HDF5
path = "test_writer.nc"

x = np.random.rand(100, 100)
x = x.astype(np.float32)
x[1, :] = np.nan

print(x)
with Dataset(path, "w", format="NETCDF4") as fp:
    HDF5.write(fp, x, "x", ("t", "p"), "uint16", scale_factor=0.0001)

y = HDF5Reader(path).read("x")[:]
print(y)
print(y.data)
print(y.fill_value)