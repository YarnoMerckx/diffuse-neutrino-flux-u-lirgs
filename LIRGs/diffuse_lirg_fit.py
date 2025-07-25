import numpy as np

from scipy.optimize import curve_fit
from scipy.integrate import quad

from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import astropy.constants as const



def Rp(E, Emin, Emax, s):
    """
    Calculates the energy normalization factor Rp for a power-law spectrum for plotting purposes.

    Parameters:
    - E (float): Energy at which the rate is evaluated.
    - Emin (float): Minimum energy of the range.
    - Emax (float): Maximum energy of the range.
    - s (float): Spectral index of the power-law.

    Returns:
    - Rp (float): Normalization factor depending on the spectral index.
    """
    if s == 2:
        # Special case: integral of E^-2 is logarithmic
        return np.log(Emax / Emin)
    else:
        # General case: integral over E^-s normalized at energy E
        numerator = Emin**(-s + 2) - Emax**(-s + 2)
        denominator = s - 2
        return (numerator / denominator) * E**(s - 2)

def nuflux_fit(E, eta, alpha, xi, fpp):
    """
    Computes a modeled neutrino flux based on IR luminosity and fitting parameters.

    Parameters:
    - E (float or array): Neutrino energy [GeV].
    - eta (float): CR acceleration efficiency.
    - alpha (float): Proton spectral index.
    - xi (float): Redshift evolution factor.
    - fpp (float): Proton-to-pion conversion efficiency.

    Returns:
    - Neutrino flux [GeV-1 cm-2 s-1 sr-1]
    """
    
    def generation_rate_diff_fit(E, alpha, eta):
        """
        Internal helper function to compute Q(E) assuming Eν = ECR / 20.
        Uses a fixed total IR luminosity for DL_compl = 75 Mpc.
        """
        Emin = E[0] * 20
        Emax = E[-1] * 20

        if alpha == 2:
            R = Rp(E,Emin,Emax,alpha)
        else:
            R = Rp(E,Emin,Emax,alpha)
            #((Emin ** (-alpha + 2) - Emax ** (-alpha + 2)) / (alpha - 2)) * E ** (alpha - 2)

        # Hardcoded total Q_IR from DL_compl = 75 Mpc
        Qir_value = 6.2213173960085935e+47  # [erg Mpc-3 yr-1]
        return (Qir_value * eta) / R

    # Inverse Hubble time in seconds
    tH = (1 / cosmo.H(0)).to((u.Mpc * u.s) / u.Mpc)

    # Hubble distance in cm
    ctH = (const.c.to(u.cm / u.s) * tH).value

    # Convert generation rate to [GeV / (cm3 s)]
    generation_rate = (
        generation_rate_diff_fit(E, alpha, eta)
        * u.erg / (u.Mpc**3 * u.yr)
    ).to(u.GeV / (u.cm**3 * u.s)).value

    # Kpi = 0.5 for pp → π± channel (1/3 flavor factor assumed)
    return (1 / 3) * ((ctH * xi) / (4 * np.pi)) * 0.5 * fpp * generation_rate


def fit_eta_vs_fpp(
    nuflux_model,
    energy_array,
    flux_array,
    xi,
    fpp_range=np.arange(0.1, 1.01, 0.01),
    alpha_fixed=2.37,  # fixed alpha
):
    """
    Fits only eta vs. fpp using a specified neutrino flux model, with alpha fixed.

    Parameters:
    - nuflux_model (function): callable like `nuflux_fit(E, eta, alpha, fpp, xi)`
    - energy_array (array): Energy values [GeV]
    - flux_array (array): Measured flux × E² (to fit to)
    - fpp_range (array): Values of proton-to-pion conversion efficiency to test.
    - xi (float): Redshift evolution factor (xi_z)
    - alpha_fixed (float): Fixed spectral index alpha
    - p0 (tuple): Initial guess for eta only (one parameter)

    Returns:
    - fpp_vals (array): Array of fpp values used
    - eta_fit (array): Best-fit η values for each fpp
    """

    eta_fit = []

    for fpp_val in fpp_range:
        try:
            popt, _ = curve_fit(
                lambda E, eta: nuflux_model(E, eta, alpha_fixed, fpp_val, xi),
                energy_array,
                flux_array,
            )
            eta_fit.append(popt[0])
        except RuntimeError:
            print('Fit failed for fpp =', fpp_val)
            eta_fit.append(np.nan)

    return fpp_range, np.array(eta_fit)
