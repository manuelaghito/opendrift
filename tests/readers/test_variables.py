import numpy as np
import pyproj
from . import *
from datetime import datetime, timedelta
from opendrift import test_data_folder as tdf
from opendrift.readers import reader_netCDF_CF_generic
from opendrift.readers import reader_constant
from opendrift.models.physics_methods import wind_drift_factor_from_trajectory
from opendrift.models.oceandrift import OceanDrift

from opendrift.readers.basereader.variables import ReaderDomain

def test_get_variables_along_trajectory_and_wind_drift_factor_from_trajectory():
    o = OceanDrift(loglevel=50)
    o.add_readers_from_list([tdf +
        '16Nov2015_NorKyst_z_surface/norkyst800_subset_16Nov2015.nc',
        tdf +
        '16Nov2015_NorKyst_z_surface/arome_subset_16Nov2015.nc'], lazy=False)

    o.env.finalize()
    t = o.env.get_variables_along_trajectory(variables=['x_sea_water_velocity', 'y_sea_water_velocity', 'x_wind', 'y_wind'],
            lons=np.array([3.5, 4, 4.5]), lats=np.array([59.7, 60, 60.3]),
            times=[o.env.readers[list(o.env.readers)[0]].start_time+i*timedelta(hours=3) for i in range(3)])
    np.testing.assert_array_almost_equal(t['x_sea_water_velocity'], [-0.078685, -0.106489, -0.058386])
    np.testing.assert_array_almost_equal(t['x_wind'], [-8.308249, -13.063459, -11.09289])

    wdf, azimuth = wind_drift_factor_from_trajectory(t)
    np.testing.assert_array_almost_equal(wdf, [0.27189012, 0.20492421], 5)
    np.testing.assert_array_almost_equal(azimuth, [73.0112213, 82.39749185], 5)

def test_modulate_longitude_360():
    class R360(ReaderDomain):
        xmin = 0
        xmax = 340
        ymin = -80
        ymax = 80

        def __init__(self):
            self.proj4 = '+proj=lonlat +ellps=WGS84'
            self.crs = pyproj.CRS(self.proj4)
            self.proj = pyproj.Proj(self.proj4)
            super().__init__()

    r = R360()
    lons = np.linspace(0, 300, 100)

    assert (r.modulate_longitude(lons) == lons).all()

    lons = np.array([-180, -90])
    assert (r.modulate_longitude(lons) == np.array([360-180, 360-90])).all()

    lons = np.array([0, 90])
    assert (r.modulate_longitude(lons) == np.array([0, 90])).all()

    lons = np.array([100, 180])
    assert (r.modulate_longitude(lons) == np.array([100, 180])).all()

    lons = np.array([240, 350])
    assert (r.modulate_longitude(lons) == np.array([240, 350])).all()

def test_modulate_longitude_180():
    class R180(ReaderDomain):
        xmin = -150
        xmax = 180
        ymin = -80
        ymax = 80

        def __init__(self):
            self.proj4 = '+proj=lonlat +ellps=WGS84'
            self.crs = pyproj.CRS(self.proj4)
            self.proj = pyproj.Proj(self.proj4)
            super().__init__()

    r = R180()
    lons = np.linspace(-180, 150, 100)

    assert (r.modulate_longitude(lons) == lons).all()

    lons = np.array([-180, -90])
    assert (r.modulate_longitude(lons) == np.array([-180, -90])).all()

    lons = np.array([0, 90])
    assert (r.modulate_longitude(lons) == np.array([0, 90])).all()

    lons = np.array([100, 180])
    assert (r.modulate_longitude(lons) == np.array([100, -180])).all()

    lons = np.array([240])
    assert (r.modulate_longitude(lons) == np.array([-120])).all()


def test_covers_positions(test_data):
    reader_arome = reader_netCDF_CF_generic.Reader(
        test_data +
        '2Feb2016_Nordic_sigma_3d/AROME_MetCoOp_00_DEF_20160202_subset.nc')

    ts = reader_arome.get_timeseries_at_position(
        lon=12, lat=68, variables=['x_wind', 'y_wind'])

    assert len(ts['time']) == 49
    x_wind = ts['x_wind']
    assert len(x_wind) == 49

    np.testing.assert_almost_equal(x_wind[0], 2.836, 2)
    np.testing.assert_almost_equal(x_wind[-1], -0.667, 2)

def test_environment_mapping(test_data):

    # Wind from NE
    r = reader_constant.Reader({'wind_speed':5, 'wind_to_direction': 225,
                                'land_binary_mask': 0})
    o = OceanDrift(loglevel=50)
    o.set_config('general:use_auto_landmask', False)
    o.add_reader(r)
    o.seed_elements(lon=4, lat=60, time=datetime.now())
    o.run(steps=15)
    np.testing.assert_almost_equal(o.elements.lon, 3.932, 3)
    np.testing.assert_almost_equal(o.elements.lat, 59.966, 3)
    # Wind from SW
    r = reader_constant.Reader({'wind_speed':5, 'wind_to_direction': 45,
                                'land_binary_mask': 0})
    o = OceanDrift(loglevel=50)
    o.set_config('general:use_auto_landmask', False)
    o.add_reader(r)
    o.seed_elements(lon=4, lat=60, time=datetime.now())
    o.run(steps=15)
    np.testing.assert_almost_equal(o.elements.lon, 4.068, 3)
    np.testing.assert_almost_equal(o.elements.lat, 60.034, 3)

    # land_binary_mask mapped from sea_floor_depth_below_sea_level
    r = reader_netCDF_CF_generic.Reader(tdf +
            '14Jan2016_NorKyst_z_3d/NorKyst-800m_ZDEPTHS_his_00_3Dsubset.nc')
    assert 'land_binary_mask' not in r.derived_variables  # Disabled by default
    r.activate_environment_mapping('land_binary_mask_from_ocean_depth')
    assert 'land_binary_mask' in r.derived_variables

def test_lonlat_bounds_lcc(test_data_fake_ig):
    """
    Test the lonlat_bounds method for data in lcc projection
    """
    bbox = test_data_fake_ig.lonlat_bounds()
    assert np.isclose(bbox, [-109.8, 55.7, 37.8, 86.1], rtol=0.2).all()

def test_lonlat_bounds_ob_tran(test_data_fake_wind_ob_tran):
    """
    Test the lonlat_bounds method for data in ob_tran projection (rotated pole)
    """
    bbox = test_data_fake_wind_ob_tran.lonlat_bounds()
    assert np.isclose(bbox, [-28.62, 44.38, 37.01, 72.88], rtol=0.1).all()

def test_mixed_layer_thickness_from_density(tmp_path):
    import xarray as xr
    from opendrift.models.physics_methods import PhysicsMethods
    depth = np.array([0, 1, 5, 10, 20, 50, 100, 200, 500.])
    lon, lat = np.arange(0, 4.1, .5), np.arange(59, 61.1, .5)
    T = np.full((len(depth), len(lon)), 10.)
    S = np.full((len(depth), len(lon)), 35.)
    S[depth <= 5, :4] = 34  # density step between 5 and 10 m west of 1.75E
    H = np.where(lon < 1.75, 300., np.where(lon < 2.75, 40., 1000.))  # then mixed to the seafloor; then deeper than searched
    b = lambda a: np.broadcast_to(a[None, :, None, :], (2, len(depth), len(lat), len(lon))).copy()
    t0 = datetime(2023, 6, 1)
    ds = xr.Dataset(
        {'T': (('time', 'depth', 'lat', 'lon'), b(T), {'standard_name': 'sea_water_temperature'}),
         'S': (('time', 'depth', 'lat', 'lon'), b(S), {'standard_name': 'sea_water_salinity'}),
         'u': (('time', 'depth', 'lat', 'lon'), b(np.full_like(T, .1)), {'standard_name': 'x_sea_water_velocity'}),
         'H': (('lat', 'lon'), np.broadcast_to(H, (len(lat), len(lon))).copy(),
               {'standard_name': 'sea_floor_depth_below_sea_level'})},
        coords={'time': [t0, t0 + timedelta(days=1)],
                'depth': ('depth', depth, {'standard_name': 'depth', 'positive': 'down'}),
                'lat': ('lat', lat, {'standard_name': 'latitude'}),
                'lon': ('lon', lon, {'standard_name': 'longitude'})})
    ds.to_netcdf(tmp_path / 'ts.nc')
    background = reader_constant.Reader({'ocean_mixed_layer_thickness': 77, 'x_sea_water_velocity': .5,
                                         'sea_water_temperature': 10, 'sea_water_salinity': 35})

    def mld(activate, **kwargs):
        r = reader_netCDF_CF_generic.Reader(str(tmp_path / 'ts.nc'))
        if activate:
            r.activate_environment_mapping('ocean_mixed_layer_thickness_from_density', **kwargs)
        o = OceanDrift(loglevel=50)
        o.add_reader([r, background])
        o.set_config('general:use_auto_landmask', False)
        o.set_config('environment:constant:land_binary_mask', 0)
        o.set_config('drift:vertical_mixing', True)  # MLD is only needed with vertical mixing
        o.set_config('drift:profiles_depth', 100)
        o.seed_elements(lon=[1, 2, 3.5, 8], lat=[60] * 4, z=-1, time=t0)
        o.run(steps=1, time_step=600)
        return o.environment.ocean_mixed_layer_thickness, o.environment.x_sea_water_velocity

    m, u = mld(activate=False)
    np.testing.assert_array_equal(m, 77)
    m, u = mld(activate=True)
    r34, r35 = PhysicsMethods.sea_water_density(10, 34), PhysicsMethods.sea_water_density(10, 35)
    np.testing.assert_allclose(m, [5 + 5 * .03 / (r35 - r34), 40, 100, 77], rtol=1e-5)
    np.testing.assert_allclose(u, [.1, .1, .1, .5])  # currents not replaced where the layer is deeper than searched
    m, u = mld(activate=True, threshold=.5)
    np.testing.assert_allclose(m, [5 + 5 * .5 / (r35 - r34), 40, 100, 77], rtol=1e-5)

    # inputs also requested directly, with elements outside the reader (as ChemicalDrift does)
    r = reader_netCDF_CF_generic.Reader(str(tmp_path / 'ts.nc'))
    r.activate_environment_mapping('ocean_mixed_layer_thickness_from_density')
    v = ['ocean_mixed_layer_thickness', 'sea_water_temperature', 'sea_floor_depth_below_sea_level']
    env, _ = r.get_variables_interpolated(v, profiles=['sea_water_temperature'], profiles_depth=100,
                                          time=t0, lon=np.array([1, 2, 3.5, 8.]), lat=np.full(4, 60.),
                                          z=np.full(4, -1.))
    np.testing.assert_allclose(env['ocean_mixed_layer_thickness'][:3], [5 + 5 * .03 / (r35 - r34), 40, 100], rtol=1e-5)
    assert np.ma.getmaskarray(env['ocean_mixed_layer_thickness'])[3]
