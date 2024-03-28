# Write unit tests for the astropy_cosmo module
import numpy as np
from lyacosmo.astropy_cosmo import AstropyCosmo, inv_efunc
from scipy.interpolate import CubicSpline


def test_inv_efunc():
    result = inv_efunc(1.0, 0.3, 0.0, 0.7, -1)
    assert result is not None
    assert isinstance(result, float)


def test_AstropyCosmo_init():
    cosmo = AstropyCosmo(0.3, 0.7, 8e-5, -1.1, 67.36, True)
    assert cosmo.H0 == 67.36
    assert cosmo.use_h_units
    assert isinstance(cosmo.get_dist_c, CubicSpline)
    assert isinstance(cosmo.get_dist_m, CubicSpline)

    cosmo = AstropyCosmo(0.3, use_h_units=False)
    assert not cosmo.use_h_units


def test_comoving_distance_scalar():
    cosmo = AstropyCosmo(0.3, 0.7, 0.0, -1, 67.36, True)
    result = cosmo._comoving_distance_scalar(2.3)
    assert result is not None
    assert isinstance(result, float)


def test_comoving_distance():
    cosmo = AstropyCosmo(0.3, -0.7, 0.0, -0.9, 50, False)
    result = cosmo._comoving_distance([1.0, 2.0, 3.0])
    assert result is not None
    assert isinstance(result, np.ndarray)
    assert len(result) == 3
    assert np.isclose(float(cosmo.get_dist_c(2.3)), 10790.582214326228)


def test_comoving_transverse_distance():
    cosmo = AstropyCosmo(0.3, 0., 7e-5, -1, 67.36, True)
    result = cosmo._comoving_transverse_distance(1.0)
    assert result is not None
    assert isinstance(result, float)
    assert np.isclose(float(cosmo.get_dist_m(3.4)), 4698.831363985478)
