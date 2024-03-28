import numpy as np
from scipy.interpolate import interp1d
from scipy.constants import speed_of_light

SPEED_LIGHT = speed_of_light / 1000  # [km/s]


class PiccaCosmo(object):
    """This class defines the fiducial cosmology

    This class stores useful tabulated functions to transform redshifts
    to distances using the fiducial cosmology. This is done assuming a
    FLRW metric.
    """
    def __init__(self, Omega_m, Omega_k=0., Omega_r=0., w0=-1.):
        """Initializes the methods for this instance

        Args:
            Om: float - default: 0.3
                Matter density
            Ok: float - default: 0.0
                Curvature density
            Or: float - default: 0.0
                Radiation density
            wl: float - default: -1.0
                Dark energy equation of state
            H0: float - default: 100.0
                Hubble constant at redshift 0 (in km/s/Mpc)
        """

        # WARNING: This is introduced due to historical issues in how this class
        # is coded. Using H0=100 implies that we are returning the distances
        # in Mpc/h instead of Mpc. This class should be fixed at some point to
        # make what we are doing more clear.
        H0 = 100.0

        # Ignore evolution of neutrinos from matter to radiation
        Omega_de = 1. - Omega_k - Omega_m - Omega_r

        num_bins = 10000
        z_max = 10.
        delta_z = z_max / num_bins
        z = np.arange(num_bins, dtype=float) * delta_z

        hubble = H0 * np.sqrt(
            Omega_de * (1. + z)**(3*(1 + w0))
            + Omega_k * (1. + z)**2
            + Omega_m * (1. + z)**3
            + Omega_r * (1. + z)**4
        )

        r_comov = np.zeros(num_bins)
        for index in range(1, num_bins):
            r_comov[index] = (
                SPEED_LIGHT * (1 / hubble[index - 1] + 1 / hubble[index]) / 2 * delta_z
                + r_comov[index - 1]
            )

        # D_C
        self.get_disc_c = interp1d(z, r_comov)

        # dist_m here is the comoving angular diameter distance
        if Omega_k == 0.:
            dist_m = r_comov
        elif Omega_k < 0.:
            dist_m = (np.sin(H0 * np.sqrt(-Omega_k) / SPEED_LIGHT * r_comov)
                      / (H0 * np.sqrt(-Omega_k) / SPEED_LIGHT))
        elif Omega_k > 0.:
            dist_m = (np.sinh(H0 * np.sqrt(Omega_k) / SPEED_LIGHT * r_comov)
                      / (H0 * np.sqrt(Omega_k) / SPEED_LIGHT))

        self.get_hubble = interp1d(z, hubble)
        self.distance_to_redshift = interp1d(r_comov, z)

        # D_H
        self.get_dist_hubble = interp1d(z, SPEED_LIGHT/hubble)
        # D_M
        self.get_dist_m = interp1d(z, dist_m)
        # D_V
        dist_v = np.power(z * self.get_dist_m(z)**2 * self.get_dist_hubble(z), 1/3)
        self.get_dist_v = interp1d(z, dist_v)
