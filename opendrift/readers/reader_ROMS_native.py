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
#
# Copyright 2015, Knut-Frode Dagestad, MET Norway

from bisect import bisect_left, bisect_right
from datetime import datetime, timedelta
import glob as glob_module
import os
import logging
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
from netCDF4 import num2date
import xarray as xr

from opendrift.readers import open_dataset_opendrift
from opendrift.readers.basereader import vector_pairs_xy
from opendrift.readers.basereader.structured import StructuredReader
from opendrift.readers.roppy import depth


class Reader(StructuredReader):
    """
    A reader for ROMS Output files. It can take a single file, a file pattern, a URL or an xarray Dataset.

    Args:
        :param filename: A single netCDF file, a pattern of files, or a xr.Dataset. The
                         netCDF file can also be an URL to an OPeNDAP server.
        :type filename: string, xr.Dataset (required).

        :param name: Name of reader
        :type name: string, optional

        :param save_interpolator: Whether or not to save the interpolator that goes from lon/lat to x/y (calculated in structured.py)
        :type save_interpolator: bool

        :param interpolator_filename: If save_interpolator is True, user can input this string to control where interpolator is saved.
        :type interpolator_filename: Path, str, optional

        :param destagger_velocity: Average u/v from their staggered grid points onto the rho grid.
        :type destagger_velocity: bool

        :param time_buffer: If set (timedelta, hours as float, or '48h'/'2d'), only
            open multi-file ROMS his files overlapping
            [time - time_buffer, time + time_buffer]. Full timeline is still
            catalogued for coverage; files are swapped as the simulation advances.
            Ignored for a single file or xr.Dataset source.
        :type time_buffer: datetime.timedelta, float, str, optional

    Example:

    .. code::

       from opendrift.readers.reader_ROMS_native import Reader
       r = Reader("roms.nc")

    Several files can be specified by using a pattern:

    .. code::

       from opendrift.readers.reader_ROMS_native import Reader
       r = Reader("*.nc")

    An OPeNDAP URL can be used:

    .. code::

       from opendrift.readers.reader_ROMS_native import Reader
       r = Reader('https://thredds.met.no/thredds/dodsC/mepslatest/meps_lagged_6_h_latest_2_5km_latest.nc')

    A xr.Dataset can be used:

    .. code::

        from opendrift.readers.reader_ROMS_native import Reader
        ds = xr.open_dataset(filename, decode_times=False)
        r = Reader(ds)
    """

    # Set in environment.py when this reader's land mask is used as landmask
    zero_velocity_on_land = False

    def __init__(self, filename=None, name=None, gridfile=None, standard_name_mapping={},
                 save_interpolator=False, interpolator_filename=None,
                 destagger_velocity=False, time_buffer=None):
        
        self.destagger_velocity = destagger_velocity
        self._mask_rho = None
        self._mask_u = None
        self._mask_v = None
        self._zeta = None
        self._angle = None
        self.land_binary_mask = None
        self.sea_floor_depth_below_sea_level = None
        self.z_rho_tot = None
        self.s2z_A = None
        self.hc = None
        # Lazy multi-file windowing (optional). When set, only his files
        # overlapping [time - time_buffer, time + time_buffer] stay open.
        self.time_buffer = self._normalize_time_buffer(time_buffer)
        self._lazy_files = False
        self._file_catalog = None  # list[(path, t0, t1, times)]
        self._loaded_files = None
        self._loaded_times = None
        self._window_start = None
        self._window_end = None
        self._gridfile = gridfile
        self._filename_source = filename
        self._drop_non_essential_vars_pop = None

        if filename is None:
            raise ValueError('Need filename as argument to constructor')

        # Map ROMS variable names to CF standard_name, for cases where standard_name attribute is missing
        self.ROMS_variable_mapping = {
            'mask_rho': 'land_binary_mask',
            'mask_psi': 'land_binary_mask',
            'h': 'sea_floor_depth_below_sea_level',
            'zeta': 'sea_surface_height',
            'u': 'x_sea_water_velocity',
            'v': 'y_sea_water_velocity',
            'u_eastward': 'eastward_sea_water_velocity',
            'v_northward': 'northward_sea_water_velocity',
            'w': 'upward_sea_water_velocity',
            'temp': 'sea_water_temperature',
            'salt': 'sea_water_salinity',
            'uice': 'sea_ice_x_velocity',
            'vice': 'sea_ice_y_velocity',
            'aice': 'sea_ice_area_fraction',
            'hice': 'sea_ice_thickness',
            'gls': 'turbulent_generic_length_scale',
            'tke': 'turbulent_kinetic_energy',
            'AKs': 'ocean_vertical_diffusivity',
            'ln_AKs': 'ocean_vertical_diffusivity',
            'sustr': 'surface_downward_x_stress',
            'svstr': 'surface_downward_y_stress',
            'tair': 'air_temperature',
            'wspd': 'wind_speed',
            'uwnd': 'x_wind',
            'vwnd': 'y_wind',
            'uwind': 'x_wind',
            'vwind': 'y_wind',
            'Uwind': 'x_wind',
            'Vwind': 'y_wind',
            }

        # Add user provided variable mappings
        self.ROMS_variable_mapping.update(standard_name_mapping)

        # z-levels to which sigma-layers may be interpolated
        self.zlevels = np.array([
            0, -.5, -1, -3, -5, -10, -25, -50, -75, -100, -150, -200,
            -250, -300, -400, -500, -600, -700, -800, -900, -1000, -1500,
            -2000, -2500, -3000, -3500, -4000, -4500, -5000, -5500, -6000,
            -6500, -7000, -7500, -8000])

        self.name = name or 'roms native'

        def drop_non_essential_vars_pop(ds):
            dropvars = [v for v in ds.variables if v not in
                        list(self.ROMS_variable_mapping.keys()) +
                        ['ocean_time', 'time', 'bulk_time', 's_rho',
                         'Cs_r', 'Cs_rho', 'hc', 'angle', 'Vtransform', 'Vstretching']
                        and v[0:3] not in ['lon', 'lat', 'mas']]
            logger.debug('Dropping variables: %s' % dropvars)
            ds = ds.drop_vars(dropvars)
            return ds
        # Keep preprocess off the stored options dict: OpenDrift's
        # test_lazy_reader compares Reader.__dict__ entries of type dict
        # by equality, and nested local-function identity would fail.
        self._drop_non_essential_vars_pop = drop_non_essential_vars_pop
        self._open_mfdataset_options = {'chunks': {'ocean_time': 1}}

        # this is an opporunity to save interpolators to pickle to save sim time
        self.save_interpolator = save_interpolator
        self.interpolator_filename = interpolator_filename or f'{self.name}_interpolators'

        sources = self._expand_sources(filename)
        if self.time_buffer is not None and len(sources) > 1:
            self._lazy_files = True
            logger.info(
                f'ROMS lazy file window enabled: time_buffer={self.time_buffer}, '
                f'n_files={len(sources)}'
            )
            self._file_catalog = self._build_file_catalog(sources)
            # Full timeline for OpenDrift coverage / nearest_time
            all_times = []
            for _path, _f0, _f1, times in self._file_catalog:
                all_times.extend(times)
            all_times = sorted(set(all_times))
            self._all_times = all_times
            # Open initial window around the first available time
            self._open_time_window(all_times[0])
        else:
            if self.time_buffer is not None and len(sources) <= 1:
                logger.info('time_buffer ignored (single file/Dataset source)')
            self.Dataset = open_dataset_opendrift(
                source=filename,
                open_mfdataset_options=self._mfdataset_options(),
            )
            self._merge_gridfile()

        self._rename_croco_dimensions()

        if 'Vtransform' in self.Dataset.variables:
            self.Vtransform = self.Dataset.variables['Vtransform'].data  # scalar
        else:
            logger.warning('Vtransform not found, using 1')
            self.Vtransform = 1
        self.Vtransform = np.asarray(self.Vtransform)

        if 'Vstretching' in self.Dataset.variables:
            self.Vstretching = self.Dataset.variables['Vstretching'].data  # scalar
        else:
            logger.warning('Vstretching not found, using 1')
            self.Vstretching = 1
        self.Vstretching = np.asarray(self.Vstretching)

        if 's_rho' not in self.Dataset.variables:
            dimensions = 2
        else:
            dimensions = 3

        if dimensions == 3:
            # Read sigma-coordinate values
            try:
                self.sigma = self.Dataset.variables['s_rho'][:]
                self.sigma = self.sigma.data  # Extract data
            except:
                num_sigma = len(self.Dataset.dimensions['s_rho'])
                logger.warning(
                    's_rho not available in dataset, constructing from'
                    ' number of layers (%s).' % num_sigma)
                self.sigma = (np.arange(num_sigma)+.5-num_sigma)/num_sigma

            # Read sigma-coordinate transform parameters
            if 'Cs_r' in self.Dataset.variables:
                csr = 'Cs_r'  # ROMS
            elif 'Cs_rho' in self.Dataset.variables:
                csr = 'Cs_rho'  # CROCO
            else:
                csr = None

            if csr is not None:
                try:
                    self.Dataset.variables[csr].set_auto_mask(False)
                except:
                    pass
                self.Cs_r = self.Dataset.variables[csr][:]

            try:
                self.hc = self.Dataset.variables['hc'].values
            except AttributeError:
                self.hc = self.Dataset.variables['hc'][:]

            self.num_layers = len(self.sigma)
        else:
            logger.warning("2D dataset, so deleting u and v from ROMS_variable_mapping")
            self.num_layers = 1
            self.ROMS_variable_mapping['ubar'] = 'x_sea_water_velocity'
            self.ROMS_variable_mapping['vbar'] = 'y_sea_water_velocity'
            del self.ROMS_variable_mapping['u']
            del self.ROMS_variable_mapping['v']

        if 'lat_rho' in self.Dataset.variables:
            # Horizontal oordinates and directions
            self.lat = self.Dataset.variables['lat_rho'][:]
            self.lon = self.Dataset.variables['lon_rho'][:]
            self.lon = self.lon.data  # Extract, could be avoided downstream
            self.lat = self.lat.data
            if self.lat.ndim == 1:
                self.lon, self.lat = np.meshgrid(self.lon, self.lat)
                # self.angle_xi_east = 0  # this was moved to the angle property
        else:
            raise ValueError(filename + ' does not contain lon/lat '
                             'arrays, please supply a grid-file: "gridfile=<grid_file>"')

        for var in list(self.ROMS_variable_mapping):  # Remove unused variables
            if var not in self.Dataset.variables:
                del self.ROMS_variable_mapping[var]

        # Get time coverage
        if self._lazy_files:
            self.times = list(self._all_times)
        else:
            ocean_time = None
            for tv in ['ocean_time', 'time', 'bulk_time']:
                if tv in self.Dataset.variables:
                    ocean_time = self.Dataset.variables[tv]
            if ocean_time is None:
                raise ValueError('Time variable not found in ROMS file')
            self.times = self._decode_ocean_time(ocean_time)
        self.start_time = self.times[0]
        self.end_time = self.times[-1]
        if len(self.times) > 1:
            self.time_step = self.times[1] - self.times[0]
        else:
            self.time_step = None

        # s2z weights (s2z_A, s2z_C). Default True = whole-domain table
        # (stock OpenDrift). False: recompute on the fetched block every call.
        self.precalculate_s2z_coefficients = True

        # Find all variables having standard_name
        self.standard_name_mapping = {}  # Inverse and unique mapping standard_name -> variable name
        unmapped_variables = []
        for var_name in list(self.Dataset.variables):
            var = self.Dataset.variables[var_name]
            if 'standard_name' in var.attrs and var_name not in self.ROMS_variable_mapping.keys():
                standard_name = var.attrs['standard_name']
            elif var_name in self.ROMS_variable_mapping:
                standard_name = self.ROMS_variable_mapping[var_name]
            else:
                unmapped_variables.append(var_name)
                continue  # Variable cannot be mapped to standard_name
            selected_variable = var_name
            if standard_name in self.standard_name_mapping:  # We have a duplicate
                if var_name in standard_name_mapping:  # provided by user
                    selected_variable = var_name
                    discarded_variable = self.standard_name_mapping[standard_name]
                else:
                    discarded_variable = var_name
                    selected_variable = self.standard_name_mapping[standard_name]
                logger.warning(f'Duplicate variables for {standard_name}, selecting {selected_variable}, '
                            f'and discarding {discarded_variable}')
            if standard_name in self.variable_aliases:  # Mapping for aliases
                standard_name = self.variable_aliases[standard_name]
            self.standard_name_mapping[standard_name] = selected_variable

        if len(unmapped_variables) > 0:
            logger.info(f'The following variables without standard_name are discarded: {unmapped_variables}')

        # A bit hackish solution:
        # If variable names or their standard_name contain "east" or "north", 
        # these should not be rotated from xi-direction to east-direction
        self.do_not_rotate = []
        for stdname, var in self.standard_name_mapping.copy().items():
            # Renaming east-north-names to x-y, as CRS of ROMS reader is always east-north,
            # and we want to avoid rotating velocities if not neceassary
            for xvar, eastnorthvar in self.xy2eastnorth_mapping.items():
                if stdname in eastnorthvar:
                    logger.info(f'Mapping {stdname} to {xvar} to avoid unnecessary rotation')
                    self.standard_name_mapping[xvar] = var
                    del self.standard_name_mapping[stdname]
                    self.do_not_rotate.append(xvar)    

        for stdname, var in self.standard_name_mapping.copy().items():
            # Also avoid rotation of these variables
            if 'east' in var.lower() or 'east' in stdname.lower() or \
                    'north' in var.lower() or 'north' in stdname.lower():
                if stdname not in self.do_not_rotate:
                    self.do_not_rotate.append(stdname)

        if len(self.do_not_rotate)>0:
            logger.debug('The following ROMS vectors are considered east-north, and will not be rotated %s' % self.do_not_rotate)


        self.variables = list(self.standard_name_mapping)

        # Run constructor of parent Reader class
        super(Reader, self).__init__()

    @property
    def mask_rho(self):
        """Mask for the rho-points.
        
        Uses wetdry_mask_rho (which should be 3D) if available, otherwise mask_rho (2D).
        If this mask is 2D, read it in this one time and use going forward in simulation. If 3D,
        will read in parts of the mask each loop.
        """
        if self._mask_rho is None:
            if 'wetdry_mask_rho' in self.Dataset.data_vars:
                self._mask_rho = self.Dataset.variables['wetdry_mask_rho']
                logger.info("Using wetdry_mask_rho for mask_rho")
            else:
                # Read landmask for whole domain, for later re-use
                self._mask_rho = self.Dataset.variables['mask_rho'][:]
                # load in once if static mask
                if self._mask_rho.chunks is not None:  # to see if dask array
                    self._mask_rho = self._mask_rho.compute()
                logger.info("Using mask_rho for mask_rho")
        return self._mask_rho

    @property
    def mask_u(self):
        """Mask for the u-points.
        
        Uses wetdry_mask_u (which should be 3D) if available, otherwise mask_u (2D).
        If this mask is 2D, read it in this one time and use going forward in simulation. If 3D,
        will read in parts of the mask each loop.
        """
        if self._mask_u is None:
            if 'wetdry_mask_u' in self.Dataset.data_vars:
                self._mask_u = self.Dataset.variables['wetdry_mask_u']
                logger.info("Using wetdry_mask_u for mask_u")
            else:
                # Read landmask for whole domain, for later re-use
                self._mask_u = self.Dataset.variables['mask_u'][:]
                # load in once if static mask
                if self._mask_u.chunks is not None:  # to see if dask array
                    self._mask_u = self._mask_u.compute()
                logger.info("Using mask_u for mask_u")
        return self._mask_u

    @property
    def mask_v(self):
        """Mask for the v-points.
        
        Uses wetdry_mask_v (which should be 3D) if available, otherwise mask_v (2D).
        If this mask is 2D, read it in this one time and use going forward in simulation. If 3D,
        will read in parts of the mask each loop.
        """
        if self._mask_v is None:
            if 'wetdry_mask_v' in self.Dataset.data_vars:
                self._mask_v = self.Dataset.variables['wetdry_mask_v']
                logger.info("Using wetdry_mask_v for mask_v")
            else:
                # Read landmask for whole domain, for later re-use
                self._mask_v = self.Dataset.variables['mask_v'][:]
                # load in once if static mask
                if self._mask_v.chunks is not None:  # to see if dask array
                    self._mask_v = self._mask_v.compute()
                logger.info("Using mask_v for mask_v")
        return self._mask_v
    
    @property
    def zeta(self):
        """Sea surface height."""
        if self._zeta is None:
            if 'zeta' in self.Dataset.data_vars:
                self._zeta = self.Dataset.variables['zeta']
                logger.info("Using zeta for sea surface height")
            else:
                self._zeta = np.zeros(self.mask_rho.shape)
                logger.info("No zeta found, using 0 array for sea surface height")
        return self._zeta        
    
    @property
    def angle(self):
        """Grid angle if curvilinear."""
        if self._angle is None:
            if 'lat_rho' in self.Dataset.variables and self.lat.ndim == 1:
                self._angle = 0
            elif 'angle' in self.Dataset.data_vars:
                self._angle = self.Dataset.variables['angle']
                logger.info("Using angle from Dataset.")
            # else:
            #     self._angle = 0
            #     logger.warning("No angle found, using 0 integer for angle.")

            if self._angle is None:
                raise ValueError('No angle between xi and east found')
        return self._angle


    @staticmethod
    def _normalize_time_buffer(time_buffer):
        """Accept timedelta, hours (int/float), or strings like ``'48h'`` / ``'2d'``.

        Returns a positive ``timedelta``, or ``None`` if ``time_buffer`` is None.
        """
        if time_buffer is None:
            return None
        if isinstance(time_buffer, timedelta):
            if time_buffer.total_seconds() <= 0:
                raise ValueError('time_buffer must be positive')
            return time_buffer
        if isinstance(time_buffer, (int, float)):
            if time_buffer <= 0:
                raise ValueError('time_buffer must be positive')
            return timedelta(hours=float(time_buffer))
        if isinstance(time_buffer, str):
            s = time_buffer.strip().lower()
            if s.endswith('h'):
                return timedelta(hours=float(s[:-1]))
            if s.endswith('d'):
                return timedelta(days=float(s[:-1]))
            return timedelta(hours=float(s))
        raise TypeError(f'Unsupported time_buffer type: {type(time_buffer)}')

    @staticmethod
    def _expand_sources(filename):
        """Turn filename / glob / list / Dataset into a flat source list.

        Used to decide whether lazy ``time_buffer`` mode applies (needs more
        than one file path). Globs are expanded and sorted.
        """
        if isinstance(filename, xr.Dataset):
            return [filename]
        if isinstance(filename, (list, tuple)):
            return list(filename)
        if isinstance(filename, str) and any(ch in filename for ch in '*?['):
            return sorted(glob_module.glob(filename))
        return [filename]

    @staticmethod
    def _is_remote_source(path):
        """True for THREDDS / OPeNDAP / HTTP(S) URLs (not local files).

        Remote sources skip the on-disk time-index cache and are catalogued
        sequentially (OPeNDAP is not process-pool friendly).
        """
        if not isinstance(path, str):
            return False
        p = path.strip().lower()
        if p.startswith('file://'):
            return False
        return (
            p.startswith('http://')
            or p.startswith('https://')
            or p.startswith('dods://')
            or p.startswith('dap://')
            or p.startswith('opendap://')
        )

    @staticmethod
    def _times_to_pydatetime(vals):
        """Normalize netCDF/xarray/cftime time values to a list of datetime."""
        if np.ndim(vals) == 0:
            vals = [vals.item() if hasattr(vals, 'item') else vals]
        else:
            vals = list(vals)
        out = []
        for t in vals:
            if isinstance(t, datetime):
                out.append(t)
            elif hasattr(t, 'year') and hasattr(t, 'month'):
                # cftime -> datetime (ROMS usually standard calendar)
                try:
                    out.append(datetime(
                        int(t.year), int(t.month), int(t.day),
                        int(getattr(t, 'hour', 0)),
                        int(getattr(t, 'minute', 0)),
                        int(getattr(t, 'second', 0)),
                    ))
                except Exception:
                    out.append(pd.Timestamp(str(t)).to_pydatetime())
            else:
                out.append(pd.Timestamp(t).to_pydatetime())
        return out

    @staticmethod
    def _read_file_times_xarray(path):
        """Read ocean_time/time/bulk_time via xarray (remote / OPeNDAP)."""
        with xr.open_dataset(path, decode_times=True) as ds:
            for tv in ['ocean_time', 'time', 'bulk_time']:
                if tv in ds.variables:
                    vals = pd.to_datetime(ds[tv].values).to_pydatetime()
                    return Reader._times_to_pydatetime(vals)
        raise ValueError(f'No time variable in {path}')

    @staticmethod
    def _read_file_times_netcdf4(path):
        """Read only the time variable from a local NetCDF file (lightweight)."""
        from netCDF4 import Dataset as NCDataset
        with NCDataset(path, 'r') as ds:
            for tv in ['ocean_time', 'time', 'bulk_time']:
                if tv not in ds.variables:
                    continue
                var = ds.variables[tv]
                raw = np.asarray(var[:]).ravel()
                if np.issubdtype(raw.dtype, np.datetime64):
                    return Reader._times_to_pydatetime(
                        pd.to_datetime(raw).to_pydatetime()
                    )
                units = getattr(var, 'units', None)
                if units is None:
                    raise ValueError(f'Time variable {tv} in {path} has no units')
                units = Reader._croco_time_units(units)
                calendar = getattr(var, 'calendar', 'standard')
                decoded = num2date(
                    raw, units=units, calendar=calendar,
                    only_use_cftime_datetimes=False,
                    only_use_python_datetimes=True,
                )
                return Reader._times_to_pydatetime(decoded)
        raise ValueError(f'No time variable in {path}')

    @staticmethod
    def _read_file_times(path):
        """Return sorted simulation times covered by one ROMS source.

        Local files use netCDF4 (time var only). Remote URLs use xarray.
        """
        if Reader._is_remote_source(path):
            return Reader._read_file_times_xarray(path)
        return Reader._read_file_times_netcdf4(path)

    @staticmethod
    def _catalog_cache_path():
        """Path of the JSON time-index cache (override with OPENDRIFT_ROMS_TIME_INDEX)."""
        env = os.environ.get('OPENDRIFT_ROMS_TIME_INDEX')
        if env:
            return env
        cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'opendrift')
        return os.path.join(cache_dir, 'roms_file_times.json')

    @staticmethod
    def _file_identity(path):
        """Return (realpath, mtime_ns, size) used as cache key for local files."""
        st = os.stat(path)
        mtime_ns = getattr(st, 'st_mtime_ns', int(st.st_mtime * 1e9))
        return os.path.realpath(path), int(mtime_ns), int(st.st_size)

    @staticmethod
    def _load_time_index(cache_path):
        """Load cached per-file time lists, or ``{}`` if missing/unreadable."""
        import json
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    @staticmethod
    def _save_time_index(cache_path, index):
        """Atomically write the per-file time-index JSON cache."""
        import json
        try:
            os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
            tmp = cache_path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(index, f)
            os.replace(tmp, cache_path)
        except Exception as e:
            logger.debug(f'ROMS time index not saved ({e})')

    def _build_file_catalog(self, sources):
        """Build ``[(path, t0, t1, times), ...]`` for lazy multi-file windowing.

        Reads each file's time axis once (sequential; skip workers are
        daemonic and cannot spawn a process pool), with an optional disk
        cache keyed by path identity. Required before opening a sliding
        ``time_buffer`` window over many ROMS his files.
        """
        from datetime import datetime as _dt

        for path in sources:
            if isinstance(path, xr.Dataset):
                raise ValueError('time_buffer lazy mode requires file paths, not a Dataset')

        cache_path = self._catalog_cache_path()
        index = self._load_time_index(cache_path)
        catalog = []
        to_read = []
        n_cached = 0

        for path in sources:
            if self._is_remote_source(path):
                to_read.append(path)
                continue
            try:
                real, mtime_ns, size = self._file_identity(path)
            except OSError:
                to_read.append(path)
                continue
            hit = index.get(real)
            if (
                isinstance(hit, dict)
                and hit.get('mtime_ns') == mtime_ns
                and hit.get('size') == size
                and hit.get('times')
            ):
                times = [_dt.fromisoformat(s) for s in hit['times']]
                catalog.append((path, times[0], times[-1], times))
                n_cached += 1
                logger.debug(
                    f'ROMS catalog {path}: {times[0]} -> {times[-1]} '
                    f'({len(times)} steps, cached)'
                )
            else:
                to_read.append(path)

        t0 = datetime.now()
        results = [(path, self._read_file_times(path)) for path in to_read]
        if to_read:
            logger.info(
                f'ROMS catalog reading {len(to_read)} files sequentially '
                f'({n_cached} cached)'
            )

        for path, times in results:
            catalog.append((path, times[0], times[-1], times))
            logger.debug(
                f'ROMS catalog {path}: {times[0]} -> {times[-1]} ({len(times)} steps)'
            )
            if not self._is_remote_source(path):
                try:
                    real, mtime_ns, size = self._file_identity(path)
                    index[real] = {
                        'mtime_ns': mtime_ns,
                        'size': size,
                        'times': [t.isoformat() for t in times],
                    }
                except OSError:
                    pass

        if results:
            self._save_time_index(cache_path, index)

        catalog.sort(key=lambda c: c[1])
        elapsed = (datetime.now() - t0).total_seconds()
        logger.info(
            f'ROMS catalog ready: {len(catalog)} files in {elapsed:.1f}s '
            f'({n_cached} cached, {len(to_read)} read)'
        )
        return catalog


    def _merge_gridfile(self):
        """Merge ``gridfile`` variables into the currently open Dataset."""
        if self._gridfile is None:
            return
        gf = xr.open_dataset(self._gridfile)
        for var in gf:
            if var in self.Dataset:
                logger.warning(f'Skipping duplicate variable {var} from grdfile')
                gf = gf.drop_vars(var)
        self.Dataset = xr.merge([self.Dataset, gf], compat='override')

    def _rename_croco_dimensions(self):
        """Fix wrongly named Croco dimensions (58cfaa56). Re-run after each lazy window open."""
        for var in self.Dataset:
            dims = self.Dataset[var].dims
            if 'eta_rho' in dims and 'xi_u' in dims:
                logger.warning(f'Wrongly named dimensions of {var} in Croco output: ' +
                               'Renaming dimensions (eta_rho, xi_u) to (eta_u, xi_u)')
                self.Dataset[var] = self.Dataset[var].rename({'eta_rho': 'eta_u'})
            elif 'eta_v' in dims and 'xi_rho' in dims:
                logger.warning(f'Wrongly named dimensions of {var} in Croco output: ' +
                               'Renaming dimensions (eta_v, xi_rho) to (eta_v, xi_v)')
                self.Dataset[var] = self.Dataset[var].rename({'xi_rho': 'xi_v'})

    @staticmethod
    def _croco_time_units(time_units):
        """Map non-CF CROCO units ('second' / 'days') to a decodeable string (089d28d1)."""
        if time_units == 'second':
            logger.info('Ocean time given as seconds relative to start '
                        'Setting artifical start time of 1 Jan 2000.')
            return 'seconds since 2000-01-01 00:00:00'
        if time_units == 'days':
            logger.info('Ocean time given as days relative to start '
                        'Setting artifical start time of 1 Jan 2000.')
            return 'days since 2000-01-01 00:00:00'
        return time_units

    @staticmethod
    def _decode_ocean_time(ocean_time):
        """Decode ROMS/CROCO time to a list of datetime (CF or the 089d28d1 hack)."""
        if np.issubdtype(ocean_time.dtype, np.datetime64):
            times = pd.to_datetime(ocean_time).to_pydatetime()
        else:
            time_units = ocean_time.attrs['units']
            time_units = Reader._croco_time_units(time_units)
            times = num2date(ocean_time[:], time_units)
        if isinstance(times, np.ndarray):
            times = list(times)
        return times

    def _select_files_for_window(self, center_time):
        """Return his files overlapping ``[center_time ± time_buffer]``.

        Also returns the window bounds ``(t0, t1)``. If none overlap, falls back
        to the single nearest file in the catalogue.
        """
        t0 = center_time - self.time_buffer
        t1 = center_time + self.time_buffer
        selected = []
        for path, f0, f1, _times in self._file_catalog:
            if f1 < t0 or f0 > t1:
                continue
            selected.append(path)
        if not selected:
            path, f0, f1, _times = min(
                self._file_catalog,
                key=lambda c: abs((c[1] - center_time).total_seconds()),
            )
            selected = [path]
            logger.warning(
                f'No ROMS files in buffer around {center_time}; using nearest {path}'
            )
        return selected, t0, t1

    def _mfdataset_options(self):
        """Options for ``open_dataset_opendrift`` / ``open_mfdataset``.

        ``preprocess`` is attached here (not stored on the options dict) so
        Reader attribute comparison in tests stays stable across instances.
        """
        opts = dict(self._open_mfdataset_options)
        if self._drop_non_essential_vars_pop is not None:
            opts['preprocess'] = self._drop_non_essential_vars_pop
        return opts

    def _open_time_window(self, center_time):
        """Open (or reuse) the mfdataset covering ``center_time ± time_buffer``.

        Closes the previous Dataset when the selected file set changes and
        clears cached sigma→z arrays that belong to the old window.
        """
        selected, _, _ = self._select_files_for_window(center_time)
        if self._loaded_files == selected:
            return
        logger.info(
            f'ROMS opening {len(selected)}/{len(self._file_catalog)} files '
            f'for window around {center_time} '
            f'(buffer={self.time_buffer})'
        )
        old = getattr(self, 'Dataset', None)
        self.Dataset = open_dataset_opendrift(
            source=selected if len(selected) > 1 else selected[0],
            open_mfdataset_options=self._mfdataset_options(),
        )
        self._merge_gridfile()
        self._rename_croco_dimensions()
        if old is not None:
            try:
                old.close()
            except Exception:
                pass
        self._loaded_files = list(selected)
        ocean_time = None
        for tv in ['ocean_time', 'time', 'bulk_time']:
            if tv in self.Dataset.variables:
                ocean_time = self.Dataset.variables[tv]
                break
        self._loaded_times = list(self._decode_ocean_time(ocean_time))
        self._window_start = self._loaded_times[0]
        self._window_end = self._loaded_times[-1]
        self._clear_s2z_cache()
        self._zeta = None
        if self._mask_rho is not None and getattr(self._mask_rho, 'ndim', 2) == 3:
            self._mask_rho = None
        if self._mask_u is not None and getattr(self._mask_u, 'ndim', 2) == 3:
            self._mask_u = None
        if self._mask_v is not None and getattr(self._mask_v, 'ndim', 2) == 3:
            self._mask_v = None
        logger.info(
            f'ROMS window now {self._window_start} -> {self._window_end} '
            f'({len(self._loaded_times)} steps)'
        )

    def _ensure_time_window(self, time):
        """Swap open his files when ``time`` leaves the current buffered window.

        Called from ``get_variables`` in lazy multi-file mode. Keeps a small
        margin so the open Dataset still covers time_before / time_after.
        """
        if not self._lazy_files:
            return
        margin = self.time_step if self.time_step is not None else timedelta(hours=1)
        if (
            self._window_start is not None
            and self._window_end is not None
            and time >= self._window_start + margin
            and time <= self._window_end - margin
        ):
            return
        self._open_time_window(time)

    def _local_indx_time(self, nearest_time):
        """Map a global timeline instant to an index in the open Dataset.

        Needed because ``self.times`` spans the full catalogue while the open
        Dataset only holds the current ``time_buffer`` window.
        """
        for i, t in enumerate(self._loaded_times):
            if t == nearest_time:
                return i
            try:
                if abs((t - nearest_time).total_seconds()) < 1.0:
                    return i
            except Exception:
                pass
        i = int(np.argmin([abs((t - nearest_time).total_seconds()) for t in self._loaded_times]))
        logger.warning(
            f'ROMS local time index fallback: wanted {nearest_time}, '
            f'using {self._loaded_times[i]}'
        )
        return i

    def _destagger_axis(self, par):
        """'x' for u, 'y' for v (staggered west/south of rho), else None"""
        if not self.destagger_velocity:
            return None
        if par not in ('x_sea_water_velocity', 'y_sea_water_velocity'):
            return None
        varname = self.standard_name_mapping[par]
        # e.g. u_eastward/v_northward are already rho-grid
        return {'u': 'x', 'ubar': 'x', 'v': 'y', 'vbar': 'y'}.get(varname)

    def _destagger_velocity_component(self, par, raw_shift, indxTime, indy_shift,
                                      indx_shift, inds, z_rho_shift, z_target, s2z=None):
        """u/v (or ubar/vbar) at the other face of each rho cell, land as NaN

        s2z: cached (A, C) at the shifted cells, else multi_zslice with z_rho_shift
        """
        var = self.Dataset.variables[self.standard_name_mapping[par]]
        if var.ndim == 4 and len(np.atleast_1d(inds)) > 1:
            # same sigma-to-z as get_variables, at the shifted index
            if s2z is not None:
                A, C = s2z
                F = np.asarray(raw_shift).reshape(raw_shift.shape[0], -1)
                I = np.arange(F.shape[1])
                raw_shift = ((1-A)*F[(C-1, I)] + A*F[(C, I)]).reshape(
                    (A.shape[0],) + raw_shift.shape[1:])
            else:
                raw_shift, _, _, _, _ = depth.multi_zslice(raw_shift, z_rho_shift, z_target)
            raw_shift[raw_shift > 1e+9] = np.nan

        mask_attr = 'mask_u' if self._destagger_axis(par) == 'x' else 'mask_v'
        mask_obj = getattr(self, mask_attr)  # mask_u/mask_v: static (2D) or wetdry (3D)
        ixy_shift = (indy_shift, indx_shift)
        mask_shift = np.asarray(mask_obj[(indxTime,) + ixy_shift if mask_obj.ndim == 3
                                         else ixy_shift])
        if mask_shift.min() == 0:
            raw_shift = np.where(mask_shift == 0, np.nan, raw_shift)

        return np.asarray(raw_shift, dtype=float)

    def get_variables(self, requested_variables, time=None,
                      x=None, y=None, z=None, testing=False):
        start_time = datetime.now()
        requested_variables, time, x, y, z, outside = self.check_arguments(
            requested_variables, time, x, y, z)

        # land_binary_mask should be based on the rho grid
        if 'land_binary_mask' in requested_variables and self.land_binary_mask is None:
            self.land_binary_mask = 1 - self.mask_rho

        if 'sea_floor_depth_below_sea_level' in requested_variables and self.sea_floor_depth_below_sea_level is None:
            self.sea_floor_depth_below_sea_level = self.Dataset.variables['h'][:]

        # If one vector component is requested, but not the other
        # we must add the other for correct rotation
        for vector_pair in vector_pairs_xy:
            if (vector_pair[0] in requested_variables and
                vector_pair[1] not in requested_variables):
                requested_variables.extend([vector_pair[1]])
            if (vector_pair[1] in requested_variables and
                vector_pair[0] not in requested_variables):
                requested_variables.extend([vector_pair[0]])

        nearestTime, dummy1, dummy2, indxTime, dummy3, dummy4 = \
            self.nearest_time(time)

        if self._lazy_files:
            self._ensure_time_window(time)
            indxTime = self._local_indx_time(nearestTime)

        variables = {}

        if z is None:
            z = np.atleast_1d(0)

        # Find horizontal indices corresponding to requested x and y
        if self.clipped is not None:
            clipped = self.clipped
        else: clipped = 0
        indx = np.floor((x-self.xmin)/self.delta_x).astype(int) + clipped
        indy = np.floor((y-self.ymin)/self.delta_y).astype(int) + clipped
        indx[outside] = 0  # To be masked later
        indy[outside] = 0
        indx_el = indx.copy()
        indy_el = indy.copy()
        # Adding buffer, to cover also future positions of elements
        buffer = self.buffer
        # Avoiding the last pixel in each dimension, since there are
        # several grids which are shifted (rho, u, v, psi)
        xi1 = int(np.max([0, indx.min()-buffer]))
        xi2 = int(np.min([indx.max()+buffer, self.lon.shape[1]-1]))
        yi1 = int(np.max([0, indy.min()-buffer]))
        yi2 = int(np.min([indy.max()+buffer, self.lon.shape[0]-1]))
        indx = np.arange(xi1, xi2)
        indy = np.arange(yi1, yi2)
        # One more cell west/south, for the other face of u/v (de-staggering)
        ext = 1 if self.destagger_velocity else 0
        indx_e = np.arange(max(xi1 - ext, 0), xi2)
        indy_e = np.arange(max(yi1 - ext, 0), yi2)
        indx_shift = np.clip(indx - 1, 0, None)
        indy_shift = np.clip(indy - 1, 0, None)

        # define indices; slices, since index arrays are much slower through xarray/dask
        sx, sy = slice(xi1, xi2), slice(yi1, yi2)
        sx_e, sy_e = slice(int(indx_e[0]), xi2), slice(int(indy_e[0]), yi2)
        ixy = (sy, sx)
        itxy = (indxTime, sy, sx)
        
        # use these same indices for all mask subsetting since if one is
        # 3D they all should be
        if self.mask_rho.ndim == 2:
            imask = ixy
        elif self.mask_rho.ndim == 3:
            imask = itxy

        # Find depth levels covering all elements
        if z.min() == 0 or self.hc is None:
            inds = self.num_layers - 1  # surface layer
            variables['z'] = 0

        else:
            # Find the range of indices covering given z-values
            if self.sea_floor_depth_below_sea_level is None:
                logger.debug('Reading sea floor depth...')
                self.sea_floor_depth_below_sea_level = \
                    self.Dataset.variables['h'][:]

            if self.z_rho_tot is None:
                self._calculate_z_rho_tot(indxTime)

            H = self.sea_floor_depth_below_sea_level[sy_e, sx_e]
            zeta = np.asarray(self.zeta[(indxTime, sy_e, sx_e)])
            # Element indices relative to the extracted subset
            indx_el = np.clip(indx_el - indx.min(), 0, len(indx) - 1)
            indy_el = np.clip(indy_el - indy.min(), 0, len(indy) - 1)
            if self.precalculate_s2z_coefficients is True:
                # The cached coefficients do the sigma-to-z; depths are only
                # needed at the element columns, for the layer range below
                jy, jx = indy_el + yi1 - indy_e[0], indx_el + xi1 - indx_e[0]
                H, zeta = np.asarray(H)[jy, jx], zeta[jy, jx]
            z_rho_e = depth.sdepth(H, zeta, self.hc, self.Cs_r,
                                   Vtransform=self.Vtransform,
                                   Vstretching=self.Vstretching)

            # z_rho is positive relative to mean sea level but z is
            # 0 at the surface.
            # Transform z_rho to match convention of z.
            z_rho_e -= zeta[np.newaxis]
            if self.precalculate_s2z_coefficients is True:
                z_rho = z_rho_e
                z_rho_el = z_rho
            else:
                z_rho = z_rho_e[:, yi1 - indy_e[0]:, xi1 - indx_e[0]:].copy()

            # Check for positive values in z_rho
            # if there are any, nan them out since they are above
            # the surface and therefore the cell is dry
            # this should only come up for wet/dry simulations
            if (np.nanmax(z_rho) > 0).any():
                logger.info(f'z_rho had positive values that are now nans.')
                z_rho[z_rho>0] = np.nan

            if self.precalculate_s2z_coefficients is not True:
                z_rho_el = z_rho[:, indy_el, indx_el]

            # Loop to find the layers covering the requested z-values
            inds_min = 0
            inds_max = self.num_layers
            for i in range(self.num_layers):
                if np.min(z-z_rho_el[i]) > 0:
                    inds_min = i
                if np.max(z-z_rho_el[i]) > 0:
                    inds_max = i
            inds = range(np.maximum(0, inds_min-self.verticalbuffer),
                         np.minimum(self.num_layers,
                                    inds_max + 1 + self.verticalbuffer))
            # Determine the z-levels to which to interpolate
            zi1 = np.maximum(0, bisect_left(-np.array(self.zlevels),
                                            -z.max()) - self.verticalbuffer)
            zi2 = np.minimum(len(self.zlevels),
                             bisect_right(-np.array(self.zlevels),
                                          -z.min()) + self.verticalbuffer)
            variables['z'] = np.array(self.zlevels[zi1:zi2])
        
        def sigma_index(inds):
            return slice(inds.start, inds.stop) if isinstance(inds, range) else inds

        # define another set of indices
        itsxy = (indxTime, sigma_index(inds), sy, sx)
            
        def get_mask(mask_name, imask, masks_store):
            if mask_name in masks_store:
                mask = masks_store[mask_name]
            else:
                mask = getattr(self, mask_name)[imask]
            return mask, mask_name

        def read_block(var, index, par):
            """var[index], and for u/v the cells one west/south (or None)"""
            axis = self._destagger_axis(par)
            if axis is None:
                return var[index], None
            # Read both faces at once: one cell more along the staggered axis
            if axis == 'x':
                index_e, ax = index[:-1] + (sx_e,), -1
                i_main, i_shift = indx - indx_e[0], indx_shift - indx_e[0]
            else:
                index_e, ax = index[:-2] + (sy_e, sx), -2
                i_main, i_shift = indy - indy_e[0], indy_shift - indy_e[0]
            block = np.asarray(var[index_e])
            return np.take(block, i_main, axis=ax), np.take(block, i_shift, axis=ax)

        raw_shifted = {}  # u/v at the other face, for de-staggering
        masks_store = {}  # To store masks for various grids
        s2z_inds_finalized = False
        for par in requested_variables:
            varname = self.standard_name_mapping[par]
            var = self.Dataset.variables[varname]

            if par == 'land_binary_mask':
                variables[par] = self.land_binary_mask[imask]
            elif par == 'sea_floor_depth_below_sea_level':
                variables[par] = self.sea_floor_depth_below_sea_level[ixy]
            elif var.ndim == 2:
                variables[par] = var[ixy]
            elif var.ndim == 3:
                variables[par], raw_shifted[par] = read_block(var, itxy, par)
            elif var.ndim == 4:
                # 4D fields F are read on sigma layers, then interpolated to
                # z-levels by R = (1-A)*F[C-1] + A*F[C], where C and C-1 are
                # the sigma layers above and below the required z. With
                # precalculate_s2z_coefficients == True, C comes from the
                # whole-domain table as a full-column index, 0 = seabed to
                # num_layers-1 = surface.
                #
                # To save I/O we read F only for a subset of layers:
                # inds -> itsxy -> variables[par] -> F. So F[0] is layer
                # si_bottom, not layer 0. Before C can be used,
                # inds must contain every layer C and C-1 over
                # [zi1:zi2, yi1:yi2, xi1:xi2], so it is expanded to cover
                # [C_bottom-1 .. C_top]. C is then shifted to F numbering
                # by C = C - si_bottom (below).
                #
                # Done once, on the first 4D field; later fields reuse it.
                if not s2z_inds_finalized:
                    if (self.precalculate_s2z_coefficients is True and
                            len(np.atleast_1d(inds)) > 1):
                        si_bottom, si_top = int(min(inds)), int(max(inds))
                        self._ensure_s2z_coefficients()
                        # layers also cover the other u/v face (de-staggering)
                        C_e = np.asarray(self.s2z_C[zi1:zi2, sy_e, sx_e])
                        C = C_e[:, yi1 - indy_e[0]:, xi1 - indx_e[0]:]
                        # multi_zslice always returns finite clipped ints;
                        # this guard is only for a degenerate empty slice.
                        if C_e.size:
                            C_bottom = int(C_e.min())
                            C_top = int(C_e.max())
                            si_bottom = int(min(si_bottom, max(0, C_bottom - 1)))
                            si_top = int(max(si_top, min(self.num_layers - 1, C_top)))
                            inds = range(si_bottom, si_top + 1)
                        itsxy = (indxTime, sigma_index(inds), sy, sx)
                    if not np.isscalar(inds):
                        z_rho = z_rho[inds]
                    s2z_inds_finalized = True
                variables[par], raw_shifted[par] = read_block(var, itsxy, par)
            else:
                raise Exception('Wrong dimension of variable: ' +
                                self.ROMS_variable_mapping[par])

            # If Xarray, load in so can be used in future loop iterations too
            variables[par] = np.asarray(variables[par])            

            if par != 'land_binary_mask':

                # make sure that var has matching horizontal dimensions with the mask
                # make sure coord names also match
                if self.mask_rho.shape[-2:] == var.shape[-2:] and self.mask_rho.dims[-2:] == var.dims[-2:]:
                    mask, mask_name = get_mask("mask_rho", imask, masks_store)
                elif self.mask_u.shape[-2:] == var.shape[-2:] and self.mask_u.dims[-2:] == var.dims[-2:]:
                    mask, mask_name = get_mask("mask_u", imask, masks_store)
                elif self.mask_v.shape[-2:] == var.shape[-2:] and self.mask_v.dims[-2:] == var.dims[-2:]:
                    mask, mask_name = get_mask("mask_v", imask, masks_store)
                else:
                    raise Exception('No mask found for ' + par)

                masks_store[mask_name] = np.asarray(mask)
                mask = np.asarray(mask)

                if mask.min() == 0:
                    # Not using the fill value directly allows the mask to have more control
                    # which is necessary when using a wetdry mask that changes in time
                    # since the fill value will not cover all masked locations then.
                    variables[par][...,mask==0] = np.nan

            if var.ndim == 4:
                # Regrid from sigma to z levels
                if len(np.atleast_1d(inds)) > 1:
                    logger.debug('sigma to z for ' + varname)
                    if self.precalculate_s2z_coefficients is True:
                        self._ensure_s2z_coefficients()
                        if 'A' not in locals():
                            logger.debug('Re-using sigma2z-coefficients')
                            # Select relevant subset of full arrays
                            zle = np.arange(zi1, zi2)  # The relevant depth levels
                            A = np.asarray(self.s2z_A[zi1:zi2, yi1:yi2, xi1:xi2])
                            # Old (commit 0c734bf5): C - C.max() + N - 1 was
                            # right only when C.max() == si_top; otherwise it
                            # used the wrong sigma layers from F.
                            C = C - si_bottom
                            C[C<1] = 1  # keep C-1 >= 0 at the seabed
                            A = A.reshape(len(zle), len(indx)*len(indy))
                            C = C.reshape(len(zle), len(indx)*len(indy))
                            I = np.arange(len(indx)*len(indy))
                            ## Check
                            #dummyvar2, s2z = depth.multi_zslice(
                            #    variables[par].copy(), z_rho.copy(), variables['z'])
                            #print len(zle), variables[par].shape, 'zle, varshape'
                            #Ac,Cc,Ic,kmaxc = s2z
                            #print C, 'C'
                            #print Cc, 'Cc'
                            #print C.shape, Cc.shape
                            #if C.max() != Cc.max():
                            #    print 'WARNING!!'
                            #    import sys; sys.exit('stop')
                            kmax = len(zle)  # Must be checked. Or number of sigma-layers?
                    if 'A' not in locals():
                        logger.debug('Calculating new sigma2z-coefficients')
                        variables[par], A,C,I,kmax = depth.multi_zslice(
                            variables[par], z_rho, variables['z'])
                        # Reshaping to compare with subset of full array
                        #zle = np.arange(zi1, zi2)
                        #A = A.reshape(len(zle), len(indx), len(indy))
                        #C = C.reshape(len(zle), len(indx), len(indy))
                        #I = I.reshape(len(indx), len(indy))
                    else:
                        logger.debug('Applying sigma2z-coefficients')
                        # Re-using sigma2z koefficients:
                        F = np.asarray(variables[par])
                        Fshape = F.shape
                        N = F.shape[0]
                        M = F.size // N
                        F = F.reshape((N, M))
                        R = (1-A)*F[(C-1, I)]+A*F[(C, I)]
                        variables[par] = R.reshape((kmax,) + Fshape[1:])

                    # Nan in input to multi_zslice gives extreme values in output
                    variables[par][variables[par]>1e+9] = np.nan

            # Mask values outside domain
            variables[par] = np.ma.array(variables[par], ndmin=2, mask=False)

            # Mean of the two faces of each rho cell, ignoring land
            if raw_shifted.get(par) is not None:
                iy_s, ix_s = ((indy, indx_shift) if self._destagger_axis(par) == 'x'
                              else (indy_shift, indx))
                z_rho_shift = s2z_shift = None
                if (var.ndim == 4 and len(np.atleast_1d(inds)) > 1 and
                        self.precalculate_s2z_coefficients is True):
                    # same cached coefficients as the main face, at the shifted cells
                    ixs = np.ix_(np.arange(zi1, zi2), iy_s, ix_s)
                    C_s = np.asarray(self.s2z_C[ixs]).reshape(zi2 - zi1, -1) - si_bottom
                    C_s[C_s < 1] = 1
                    s2z_shift = (np.asarray(self.s2z_A[ixs]).reshape(zi2 - zi1, -1), C_s)
                elif (not np.isscalar(inds) and
                        self.precalculate_s2z_coefficients is not True):
                    z_rho_shift = z_rho_e[np.asarray(inds)][
                        :, iy_s - indy_e[0]][:, :, ix_s - indx_e[0]]
                shifted = self._destagger_velocity_component(
                    par, raw_shifted[par], indxTime, iy_s, ix_s, inds, z_rho_shift,
                    variables['z'], s2z_shift)
                if shifted is not None:
                    original = np.ma.filled(variables[par], np.nan).astype(float)
                    stacked = np.stack([original, shifted])
                    with np.errstate(invalid='ignore'):
                        destaggered = np.nanmean(stacked, axis=0)
                    variables[par] = np.ma.array(destaggered, ndmin=2, mask=False)

        variables['x'] = indx
        variables['y'] = indy

        variables['x'] = variables['x'].astype(np.float32)
        variables['y'] = variables['y'].astype(np.float32)
        variables['time'] = nearestTime

        # Masked u/v from NaN to 0, per component before rotation (which mixes
        # u and v), so they are not filled with the neighbour mean
        if self.zero_velocity_on_land:
            for par in ('x_sea_water_velocity', 'y_sea_water_velocity'):
                if par in variables:
                    variables[par] = np.ma.filled(variables[par], np.nan)
                    variables[par][~np.isfinite(variables[par])] = 0

        if 'x_sea_water_velocity' in variables.keys() and \
            'x_sea_water_velocity' not in self.do_not_rotate:
            if isinstance(self.angle, int):
                rad = self.angle
            else:
                rad = np.ma.asarray(self.angle[indy, indx])
            variables['x_sea_water_velocity'], \
                variables['y_sea_water_velocity'] = rotate_vectors_angle(
                    variables['x_sea_water_velocity'],
                    variables['y_sea_water_velocity'], rad)
            logger.debug('Rotated x_sea_water_velocity and y_sea_water_velocity')
        
        if 'sea_ice_x_velocity' in variables.keys() and \
            'sea_ice_x_velocity' not in self.do_not_rotate:
            if isinstance(self.angle, int):
                rad = self.angle
            else:
                rad = np.ma.asarray(self.angle[indy, indx])
            if 'sea_ice_x_velocity' in variables.keys() and \
                    'sea_ice_x_velocity' not in self.do_not_rotate:
                variables['sea_ice_x_velocity'], \
                    variables['sea_ice_y_velocity'] = rotate_vectors_angle(
                        variables['sea_ice_x_velocity'],
                        variables['sea_ice_y_velocity'], rad)
                logger.debug('Rotated sea_ice_x_velocity and sea_ice_y_velocity')
        
        if 'x_wind' in variables.keys() and 'x_wind' not in self.do_not_rotate:
            if isinstance(self.angle, int):
                rad = self.angle
            else:
                rad = np.ma.asarray(self.angle[indy, indx])
            if 'x_wind' in variables.keys() and \
                    'x_wind' not in self.do_not_rotate:
                variables['x_wind'], \
                    variables['y_wind'] = rotate_vectors_angle(
                        variables['x_wind'],
                        variables['y_wind'], rad)
                logger.debug('Rotated x_wind and y_wind')

        if 'ocean_vertical_diffusivity' in variables.keys() and 'AKs' not in self.Dataset.variables:
            variables['ocean_vertical_diffusivity'] = np.exp(variables['ocean_vertical_diffusivity'])
            logger.info("Using AKs from ln_AKs.")
        # Masking NaN
        for var in requested_variables:
            variables[var] = np.ma.masked_invalid(variables[var])

        logger.debug('Time for ROMS native reader: ' + str(datetime.now()-start_time))

        if testing:
            return variables, masks_store
        else:
            return variables


    def _calculate_z_rho_tot(self, indxTime):
        """Whole-domain sigma layer depths at the sea surface of indxTime (the first fetch of this Dataset)"""
        Htot = self.sea_floor_depth_below_sea_level
        zeta = self.zeta[indxTime]
        self.z_rho_tot = depth.sdepth(Htot, zeta, self.hc, self.Cs_r,
                                      Vtransform=self.Vtransform,
                                      Vstretching=self.Vstretching)
        # z_rho is positive relative to mean sea level but z is
        # 0 at the surface.
        # Transform z_rho to match convention of z.
        self.z_rho_tot -= np.asarray(zeta)[np.newaxis]

    def _ensure_s2z_coefficients(self):
        """Whole-domain s2z A,C from multi_zslice, once per Dataset."""
        if self.s2z_A is not None:
            return
        logger.debug('Calculating sigma2z-coefficients for whole domain')
        starttime = datetime.now()
        M = self.sea_floor_depth_below_sea_level.shape[0]
        N = self.sea_floor_depth_below_sea_level.shape[1]
        O = len(self.z_rho_tot)
        dummyvar = np.ones((O, M, N))
        dummy, self.s2z_A, self.s2z_C, self.s2z_I, self.s2z_kmax = depth.multi_zslice(
            dummyvar, self.z_rho_tot, self.zlevels)
        self.s2z_A = self.s2z_A.reshape(len(self.zlevels), M, N)
        self.s2z_C = self.s2z_C.reshape(len(self.zlevels), M, N)
        logger.info('Time: ' + str(datetime.now() - starttime))

    def _clear_s2z_cache(self):
        self.z_rho_tot = None
        self.s2z_A = None
        self.s2z_C = None


def rotate_vectors_angle(u, v, radians):
    u2 = u*np.cos(radians) - v*np.sin(radians)
    v2 = u*np.sin(radians) + v*np.cos(radians)
    return u2, v2
