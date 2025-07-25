import numpy as np 
import pandas as pd
from astropy.cosmology import Planck18 as cosmo
import astropy.constants as const
import astropy.units as u

def Rp(E, Emin, Emax, s):
    """
    Calculates the energy normalization factor Rp for a power-law spectrum.

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


def complete_dataframe(DL_compl, df):
    """
    Filters galaxies in the dataframe `df` based on a completeness distance and LIRG luminosity threshold.

    Parameters:
    - DL_compl (float): Completeness distance limit in megaparsecs (Mpc). 
                        Galaxies beyond this distance are excluded.
    - df (DataFrame): Pandas DataFrame containing galaxy data with at least the columns 
                      'D_L [Mpc]' for luminosity distance and 'log(LIR)' for infrared luminosity.

    Returns:
    - Filtered DataFrame containing only galaxies that:
        * have D_L <= DL_compl
        * have log(LIR) < 12 (i.e., are LIRGs and not ULIRGs)
    """

    # Apply both filters:
    # 1. Galaxy distance must be less than the completeness distance
    # 2. Galaxy must be a LIRG (log(LIR) < 12)
    mask = (df['D_L [Mpc]'] <= DL_compl) & (df['log(LIR)'] < 12)

    # Return the filtered DataFrame
    return df[mask]


def QIR(DL_compl, AGNcorr,df):
    """
    Computes the IR luminosity density (Q_IR) within a given completeness distance.

    Parameters:
    - DL_compl (float): Completeness distance in Mpc.
    - AGNcorr (str): 'yes' or 'no' to apply AGN luminosity correction.

    Returns:
    - List with two values:
        1. IR luminosity density [erg Mpc^-3 yr^-1]
        2. Total IR luminosity [erg yr^-1]
    """
    complete_df = complete_dataframe(DL_compl,df)

    if AGNcorr == 'no':
        # Convert log(LIR) to LIR and sum
        LIR_complete_array = 10 ** complete_df['log(LIR)'].to_numpy()
        totalIR = (np.sum(LIR_complete_array) * u.solLum).to(u.erg / u.yr)

    elif AGNcorr == 'yes':
        # Subtract AGN fraction before summing IR luminosity
        logLIR = complete_df['log(LIR)'].to_numpy()
        AGNfrac = complete_df['AGNbol'].to_numpy()
        LIR_corrected = [(1 - f) * 10 ** l for f, l in zip(AGNfrac, logLIR)]
        totalIR = (np.sum(LIR_corrected) * u.solLum).to(u.erg / u.yr)

    # Compute comoving volume within DL_compl
    Dmax = DL_compl * u.Mpc
    Volume = (4 / 3) * np.pi * Dmax ** 3

    return [(totalIR / Volume).value, totalIR.value]



def generation_rate_diff(DL_compl, E, Emin, Emax, alpha, eta, agncorr,df):
    """
    Computes the differential energy generation rate dQ/dE.

    Parameters:
    - DL_compl (float): Completeness distance [Mpc].
    - E (float): Energy at which to evaluate [GeV or eV].
    - Emin, Emax (float): Minimum and maximum energies [same units as E].
    - alpha (float): Spectral index.
    - eta (float): Efficiency factor.
    - agncorr (str): Apply AGN correction ('yes' or 'no').

    Returns:
    - Differential energy generation rate [erg Mpc-3 yr-1]
    """

    return (QIR(DL_compl, agncorr,df)[0] * eta) / Rp(E,Emin,Emax,alpha)



    
    
def nuflux(E, Emin, Emax, alpha, DL_compl, xiz, eta, fpp, channel, agncorr,df):
    """
    Computes the diffuse neutrino flux (per flavor) from the LIRG population.

    Parameters:
    - E (float): Neutrino energy at which to evaluate the flux [GeV].
    - Emin, Emax (float): Minimum and maximum proton energies [same units as E].
    - alpha (float): Proton spectral index.
    - DL_compl (float): Completeness distance in Mpc.
    - xiz (float): Redshift evolution factor (ξ_z).
    - eta (float): Energy conversion efficiency from IR to CR.
    - fpp (float): Proton-to-pion conversion efficiency.
    - channel (str): Interaction channel, 'pp' or 'pγ'.
    - agncorr (str): AGN luminosity correction: 'yes' or 'no'.

    Returns:
    - Neutrino flux per flavor: φ(E) in units of [GeV-1 cm-2 s-1 sr-1]
    """
    # Inverse Hubble parameter in seconds
    tH = (1 / cosmo.H(0)).to((u.Mpc * u.s) / u.Mpc)
    
    # c × t_H gives Hubble distance in cm
    ctH = const.c.to(u.cm / u.s) * tH  # units: cm

    # Compute generation rate in physical units: [GeV / (cm3 s)]
    generation_rate = generation_rate_diff(
        DL_compl, E, Emin, Emax, alpha, eta, agncorr,df) * (u.erg / (u.Mpc**3 * u.yr))

    # Convert from erg to GeV and Mpc3 → cm3, yr-1 → s-1
    generation_rate = generation_rate.to(u.GeV / (u.cm**3 * u.s))

    # Define charged pion energy fraction per flavor based on interaction channel
    def Kpi(channel):
        return 0.5 if channel == 'pp' else 3 / 8  # 1/2 for pp, 3/8 for pγ

    # Final diffuse flux calculation per flavor [GeV-1 cm-2 s-1 sr-1]
    flux = (1 / 3) * ((ctH * xiz) / (4 * np.pi)) * Kpi(channel) * fpp * generation_rate

    return flux.value


def normalizing_factor(xiz, R, fpp, eta_tot, Qir):
    """
    Computes the normalization factor for diffuse neutrino flux based on IR luminosity density.

    Parameters:
    - xiz (float): Redshift evolution factor (ξ_z).
    - R (float): Fraction of sources (e.g., fraction of (U)LIRGs).
    - fpp (float): Pion production efficiency.
    - eta_tot (float): Total efficiency of energy conversion from IR to CR.
    - Qir (float): Infrared luminosity density [erg Mpc-3 yr-1].

    Returns:
    - Normalization constant for flux [GeV cm-2 s-1 sr-1]
    """
    # Inverse Hubble parameter in seconds
    tH = (1 / cosmo.H(0)).to((u.Mpc * u.s) / u.Mpc)

    # Hubble distance in cm
    ctH = const.c.to(u.cm / u.s) * tH  # unit: cm

    # Convert Qir to [GeV / (cm³ s)]
    Qir_density = (Qir * u.erg / (u.Mpc**3 * u.yr)).to(u.GeV / (u.cm**3 * u.s))

    # For debugging: show intermediate values (optional, remove in final version)
    print(f"Qir converted to [erg Mpc-3 yr-1]: {(Qir * u.erg / (u.Mpc**3 * u.yr))}")
    print(f"eta_tot * Qir_density: {eta_tot * Qir_density}")

    # Final normalization factor per flavor (1/6: 1/3 flavor × 1/2 from π± decay)
    normalization = xiz * R * (1 / 6) * (ctH / (4 * np.pi * u.sr)) * eta_tot * fpp * Qir_density

    return normalization



