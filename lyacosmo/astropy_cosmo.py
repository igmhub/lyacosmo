import numpy as np
from numba import njit, float64
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.constants import speed_of_light


@njit(float64(float64, float64, float64, float64, float64))
def inv_efunc(z, Omega_m, Omega_r, Omega_k, w0):
    """Hubble parameter in wCDM + curvature

    Parameters
    ----------
    z : float
        Redshift
    Omega_m : float
        Matter fraction at z = 0
    Omega_de : float
        Dark Energy fraction at z = 0
    Returns
    -------
    float
        Hubble parameter
    """
    Omega_de = 1 - Omega_m - Omega_k - Omega_r
    de_pow = 3 * (1 + w0)
    zp = 1 + z
    return (Omega_m * zp**3 + Omega_de * zp**de_pow + Omega_k * zp**2 + Omega_r * zp**4)**(-0.5)


class AstropyCosmo:
    """Class for cosmological computations based on astropy cosmology.

    Attributes
    ----------
    config: configparser.SectionProxy
    Parsed options to build cosmology

    use_hunits: bool
    If True, do the computation in h^-1Mpc. Otherwise, do it in Mpc
    """
    def __init__(
            self, Omega_m, Omega_k=0., Omega_r=0., w0=-1, H0=67.36,
            use_h_units=False, redshift_grid=None
    ):
        self.use_h_units = use_h_units
        self.H0 = H0
        self._hubble_distance = speed_of_light / 1000 / H0
        self._Omega_k = Omega_k

        # Omega_m, Omega_r, Omega_k, w
        self._inv_efunc_args = (Omega_m, Omega_r, Omega_k, w0)

        if redshift_grid is None:
            z = np.linspace(0, 10, 10000)
        else:
            z = redshift_grid

        comoving_distance = self._comoving_distance(z)
        comoving_transverse_distance = self._comoving_transverse_distance(z)

        if self.use_h_units:
            comoving_distance *= H0 / 100
            comoving_transverse_distance *= H0 / 100

        # D_C, D_M
        self.get_dist_c = interp1d(z, comoving_distance)
        self.get_dist_m = interp1d(z, comoving_transverse_distance)
        self.get_FAP = interp1d(z, comoving_transverse_distance / comoving_distance)

    def _comoving_distance_scalar(self, z):
        """Compute integral of inverse efunc for a scalar input redshift

        Parameters
        ----------
        z : float
            Target redshift

        Returns
        -------
        float
            Integral of inverse efunc between redshift 0 and input redshift
        """
        return quad(inv_efunc, 0, z, args=self._inv_efunc_args)[0]

    def _comoving_distance(self, z):
        """Compute comoving distance to target redshifts

        Parameters
        ----------
        z : float or array
            Target redshifts

        Returns
        -------
        float or array
            Comoving distances between redshift 0 and input redshifts
        """
        if isinstance(z, (list, tuple, np.ndarray)):
            return self._hubble_distance * np.array([self._comoving_distance_scalar(z_scalar)
                                                     for z_scalar in z])
        else:
            return self._hubble_distance * self._comoving_distance_scalar(z)

    def _comoving_transverse_distance(self, z):
        """Compute comoving transverse distance to target redshifts

        Parameters
        ----------
        z : float or array
            Target redshifts

        Returns
        -------
        float or array
            Comoving transverse distances between redshift 0 and input redshifts
        """
        dc = self._comoving_distance(z)
        if self._Omega_k == 0:
            return dc

        sqrt_Ok0 = np.sqrt(abs(self._Omega_k))
        dh = self._hubble_distance
        if self._Omega_k > 0:
            return dh / sqrt_Ok0 * np.sinh(sqrt_Ok0 * dc / dh)
        else:
            return dh / sqrt_Ok0 * np.sin(sqrt_Ok0 * dc / dh)
