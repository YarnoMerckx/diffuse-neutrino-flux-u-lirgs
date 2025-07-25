import pandas as pd
import numpy as np

#from astropy.cosmology import Planck18 as cosmo
import astropy.constants as const
import astropy.units as u
from astropy import cosmology
from astropy.cosmology import Planck18 as cosmo

def LoadSources(catalog_dir, 
                catalog_file,
                redshift_max=1,
                sinDec_min=-1.):
    '''
    Description
    -----------
    Function that reads out the catalog file of selected ULIRGs
    and loads in the information.
    
    
    Arguments
    ---------
    `catalog_dir`
    type        : str
    description : Directory of the catalog file.
    
    `catalog_file`
    type        : str
    description : Option to provide the name of the catalog file.
    
    `redshift_max`
    type        : float
    description : Option to specify the maximum redshift of the
                  sources that are loaded.
    
    `sinDec_min`
    type        : float
    description : Option to specfify the minimum declination of the
                  sources that are loaded.
                  
    Returns
    -------
    `ULIRGs`
    type        : dict
    description : Dictionary containing all the fields that are in the
                  catalog file. In particular it contains the fields
                  relevant for the ULIRG stacking analysis.
    '''
    
    print("\nLoading ULIRGs with z <= {} and sinDec >= {}".format(redshift_max,sinDec_min))

    # Open catalog file
    catalog = open(catalog_dir+catalog_file,"r")
    lines = catalog.readlines()

    # Lists of relevant ULIRG parameters
    # Easier to work with in first instance to load the ULIRG data
    name = []
    ra = []
    dec = []
    redshift = []
    f60 = []
    unc_f60 = []
    lum_IR = []
    catalog_info = []

    # Read out the catalog file
    # Assumes the structure of ULIRG_selection.txt
    start_readout = False
    for line in lines:
        # Skip first lines that do not contain data
        if start_readout == False and line[0][0] == "1":
            start_readout = True

        if start_readout == False:
            continue

        # Columns in the file are split with tabs
        columns = [column for column in line.split("\t") if column != ""]

        # Only load sources with a sinDec larger than `sinDec_min`
        # and with redshift smaller than `redshift_max`
        new_dec      = np.deg2rad( float(columns[3]) )
        new_redshift = float(columns[4])
        
        if np.sin( new_dec ) >= sinDec_min and new_redshift <= redshift_max:
            name.append(columns[1])
            ra.append( np.deg2rad( float(columns[2]) ) )
            dec.append( new_dec )
            redshift.append(float(columns[4]))
            f60.append(float(columns[5]))
            unc_f60.append(float(columns[6]))
            lum_IR.append(float(columns[7]))
            catalog_info.append(columns[8].split(" & "))

    catalog.close()

    # Convert the lists into arrays
    ra = np.array(ra) # [rad]
    dec = np.array(dec) # [rad]
    sinDec = np.sin(dec)
    redshift = np.array(redshift)
    f60 = np.array(f60) # [Jy]
    unc_f60 = np.array(unc_f60) # [Jy]
    lum_IR = np.array(lum_IR) # # [log10(L_sun)]

    # Calculate the luminosity distances
    # from the redshifts using astropy
    # Take the Planck 15 cosmology (H_0 = 67.7 km Mpc^-1 s^-1)
    # built in the astropy package
    distance = np.array(cosmology.Planck15.luminosity_distance(redshift)) # [Mpc]

    # Store all ULIRG parameters in one dictionary
    ULIRGs = {"name"       : name,
              "ra"         : ra,
              "dec"        : dec,
              "sinDec"     : sinDec,
              "redshift"   : redshift,
              "distance"   : distance,
              "f60"        : f60,
              "unc_f60"    : unc_f60,
              "log_lum_IR" : lum_IR,
              "catalog"    : catalog_info}

    # Return ULIRG dictionary
    print( "\nLOADED: {} ULIRGs".format(len(ra)) )
    return ULIRGs

def QIR(DL_compl,df):
    """
    Computes the infrared luminosity density (Q_IR) from a volume-limited sample of ULIRGs.

    Parameters:
    - DL_compl (float): Maximum luminosity distance for completeness (in Mpc).
    - df (DataFrame): Pandas DataFrame containing galaxy data with at least the columns 
                      'D_L [Mpc]' for luminosity distance and 'log(LIR)' for infrared luminosity.

    Returns:
    - A list containing:
        [0]: Q_IR (infrared luminosity density) in erg yr-1 cm-3
        [1]: Total infrared luminosity in erg yr-1
    """

    # Convert the ULIRGs dictionary (or structured data) into a Pandas DataFrame
    df = pd.DataFrame(data=df)

    # Select only galaxies within the completeness distance (volume-limited sample)
    volume_limited = df[(df['distance'].values <= DL_compl)]

    # Get the log(L_IR) values as a NumPy array
    logLIR_complete_array = volume_limited['log_lum_IR'].to_numpy()

    # Convert log(L_IR) to L_IR (in solar luminosities)
    LIR_array = pow(10, volume_limited['log_lum_IR']).values

    # Sum total IR luminosity and convert to erg yr-1
    totalIR = (sum(LIR_array * u.solLum)).to(u.erg /u.yr)

    # Compute volume of a sphere with radius DL_compl
    Dmax = DL_compl * u.Mpc
    Volume = (4/3) * np.pi * pow(Dmax, 3)  # Volume in Mpc3

    # Return Q_IR = totalIR / volume (in erg yr-1 cm-3), and total IR luminosity in erg/s
    return [(totalIR / Volume).value, totalIR.value]

def nu_flux(E, eta_tot, xiz, alpha, QIR, fpp, Emin, Emax):
    """
    Computes the differential neutrino flux at energy E, based on IR galaxy contribution.

    Parameters:
    - E (float): Neutrino energy (in GeV)
    - eta_tot (float): Total cosmic-ray energy injection efficiency
    - xiz (float): Redshift evolution factor (ξ_z)
    - alpha (float): Spectral index of the proton spectrum
    - QIR (float): Infrared luminosity density [erg / yr / Mpc³]
    - fpp (float): Proton-to-pion conversion efficiency
    - Emin (float): Minimum proton energy [GeV]
    - Emax (float): Maximum proton energy [GeV]

    Returns:
    - nu_flux (float): All-flavor neutrino flux at energy E [TeV / (cm² s sr)]
    """

    # Compute Rp — normalization factor for the proton injection spectrum
    if alpha == 2:
        # For alpha = 2, use logarithmic form
        Rp = np.log(Emax / Emin)
    else:
        # General form of Rp for alpha =/= 2
        Rp = ((pow(Emin, -alpha + 2) - pow(Emax, -alpha + 2)) / (alpha - 2)) * pow(E, alpha - 2)

    # Compute differential energy generation rate (per energy)
    diff = (QIR * eta_tot) / Rp  # Units: erg Mpc-3 yr

    # Compute the Hubble time: inverse of H(0), converted to seconds
    tH = (pow(cosmo.H(0), -1)).to((u.Mpc * u.s) / u.Mpc)  # Unit: s

    # Convert Hubble time to a distance in cm: ct_H = c x t_H
    ctH = (const.c).to(u.cm / u.s) * tH  # Unit: cm

    # Convert generation rate to neutrino number density: GeV cm-3 s-1
    Generation_rate_nunits = (diff * (u.erg / (pow(u.Mpc, 3) * u.yr))).to(u.GeV / (pow(u.cm, 3) * u.s))

    # Assume 100% of IR emission comes from the central engine
    factor =1

    # Compute final neutrino flux (TeV  cm-2 s-1 sr-1) and scale down by 1e-3 from GeV → TeV
    nu_flux = ((1 / 3) * ((ctH * xiz) / (4 * np.pi)) * factor * fpp * Generation_rate_nunits).value * 1e-3

    return nu_flux

