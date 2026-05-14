from ._base import TemplateData, TemplateReader

try:
    from ._hdf4 import HDF4Data, HDF4Reader
except ImportError:
    HDF4Data = None
    HDF4Reader = None

try:
    from ._hdf5 import HDF5Data, HDF5Reader
except ImportError:
    HDF5Data = None
    HDF5Reader = None

__all__ = [
    "TemplateData",
    "TemplateReader",
    "HDF4Data",
    "HDF4Reader",
    "HDF5Data",
    "HDF5Reader",
]
