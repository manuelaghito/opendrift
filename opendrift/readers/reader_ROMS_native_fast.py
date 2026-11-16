# This file is part of OpenDrift.
#
# OpenDrift is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2
#
# OpenDrift is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with OpenDrift.  If not, see <https://www.gnu.org/licenses/>.

"""reader_ROMS_native, reading time-dependent blocks with netCDF4 instead of xarray/dask.

Only in lazy multi-file mode (time_buffer): each file of the open window is also
opened with netCDF4, and a block at one time step is read from that file and
decoded with xarray's own CF decoding, so values are the same as through the
xarray Dataset. Everything else (grid, masks, static fields) still comes from
the xarray Dataset.

With zarr_dir, the blocks are read instead from <zarr_dir>/<file name>.zarr, a
copy of the time-dependent variables of each file (same raw values and
attributes), for the variables that store has.
"""

from collections.abc import Mapping
import logging
import os
logger = logging.getLogger(__name__)

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

from opendrift.readers import reader_ROMS_native


class _FastVariable:
    """Time-dependent variable: integer time index -> read from the netCDF4 file"""

    def __init__(self, xrvar, name, parts, attrs):
        self._xr = xrvar
        self._name = name
        self._parts = parts  # (netCDF4 or zarr array, index in file) per window time index
        self._attrs = attrs

    def __getattr__(self, attr):
        return getattr(self._xr, attr)

    def __getitem__(self, index):
        index = index if isinstance(index, tuple) else (index,)
        if not isinstance(index[0], (int, np.integer)):
            return self._xr[index]
        src, j = self._parts[index[0]]
        if isinstance(src, netCDF4.Variable):
            raw = src[(j,) + index[1:]]
        else:
            raw = src.oindex[(j,) + index[1:]]
        dims = tuple(f'dim{i}' for i in range(np.ndim(raw)))
        decoded = xr.conventions.decode_cf_variable(
            self._name, xr.Variable(dims, raw, dict(self._attrs)), decode_times=False)
        return decoded.values


class _FastVariables(Mapping):

    def __init__(self, ds, time_parts):
        self._ds = ds
        self._time_parts = time_parts
        self._fast = {}

    def __getitem__(self, name):
        if name in self._fast:
            return self._fast[name]
        var = self._ds.variables[name]
        if (name in self._ds.data_vars and var.dims[:1] == ('ocean_time',) and
                all(name in h.variables for h, _, _ in self._time_parts)):
            ncvar = self._time_parts[0][0].variables[name]
            attrs = {k: ncvar.getncattr(k) for k in ncvar.ncattrs()}
            self._fast[name] = _FastVariable(
                var, name, [(z[name] if z is not None and name in z else h.variables[name], j)
                            for h, z, j in self._time_parts], attrs)
            return self._fast[name]
        return var

    def __iter__(self):
        return iter(self._ds.variables)

    def __len__(self):
        return len(self._ds.variables)


class _FastDataset:
    """xarray Dataset whose .variables gives _FastVariable for time-dependent fields"""

    def __init__(self, ds, time_parts):
        self._ds = ds
        self.variables = _FastVariables(ds, time_parts)

    def __getattr__(self, attr):
        return getattr(self._ds, attr)

    def __getitem__(self, key):
        return self._ds[key]

    def __setitem__(self, key, value):
        self._ds[key] = value

    def __contains__(self, key):
        return key in self._ds

    def __iter__(self):
        return iter(self._ds)


class Reader(reader_ROMS_native.Reader):

    _nc_handles = ()

    def __init__(self, *args, zarr_dir=None, **kwargs):
        self.zarr_dir = zarr_dir
        super().__init__(*args, **kwargs)

    def _open_zarr(self, path):
        if self.zarr_dir is None:
            return None
        import zarr
        g = zarr.open_group(os.path.join(self.zarr_dir, os.path.basename(path) + '.zarr'), mode='r')
        return {name: a for name, a in g.arrays()}

    def _open_time_window(self, center_time):
        super()._open_time_window(center_time)
        if isinstance(self.Dataset, _FastDataset):
            return  # same files as before
        for h in self._nc_handles:
            h.close()
        self._nc_handles = [netCDF4.Dataset(p) for p in self._loaded_files]
        file_times = {path: times for path, _, _, times in self._file_catalog}
        by_time = {}
        for path, h in zip(self._loaded_files, self._nc_handles):
            h.set_auto_maskandscale(False)
            z = self._open_zarr(path)
            for j, t in enumerate(file_times[path]):
                by_time[round(pd.Timestamp(t).timestamp())] = (h, z, j)
        time_parts = [by_time[round(pd.Timestamp(t).timestamp())] for t in self._loaded_times]
        self.Dataset = _FastDataset(self.Dataset, time_parts)
        logger.info(f'ROMS fast read: {len(self._nc_handles)} files opened with netCDF4'
                    + (f', blocks from zarr in {self.zarr_dir}' if self.zarr_dir else ''))
