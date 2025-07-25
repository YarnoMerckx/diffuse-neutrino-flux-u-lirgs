import scipy
import numpy as np
from astropy import cosmology


def Hubble_evolution(z):
    return np.sqrt(cosmology.Planck18.Om0*(1.+z)**3. + cosmology.Planck18.Ode0)


def ULIRG_evolution(z):
    if z <= 1.:
        return (1.+z)**4.
    else:
        return 2.**4.
    

def starforming_evolution(z):
    if z <= 1.:
        return (1.+z)**(3.4)
    elif z > 1. and z <= 4.:
        return starforming_evolution(1.) * (1.+z)**(-0.3)/(2.**(-0.3))
    else:
        return starforming_evolution(4.) * (1.+z)**(-3.5)/(5.**(-3.5))
    
    
def flat_evolution(z):
    return 1.


def get_redshift_parameter(source_evolution,
                           z_max,
                           gamma):
    
    def integrand(z,gamma):
        return source_evolution(z)*(1.+z)**(-gamma) / Hubble_evolution(z)
    
    return scipy.integrate.quad(integrand,0,z_max,args=(gamma))[0]


def get_evolution(name):
    
    if name == "ULIRG":
        return ULIRG_evolution
    elif name == "starforming":
        return starforming_evolution
    elif name == "flat":
        return flat_evolution
    else:
        raise Exception("Currently unknown evolution")
        
        
def xi(g,zm,txt):
    return get_redshift_parameter(get_evolution(txt),zm,g)