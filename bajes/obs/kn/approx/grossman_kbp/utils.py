import numpy as np

# obs. the following universal constant are not expressed in SI
units_c        = 2.99792458e10     #[cm/s]
units_Msun     = 1.98855e33        #[g]
units_sigma_SB = 5.6704e-5         #[erg/cm^2/s/K^4]
units_h        = 6.6260755e-27     #[erg*s]
units_kB       = 1.380658e-16      #[erg/K]
units_pc2cm    = 3.085678e+18      #[cm/pc]

#
# Magnitude evaluation methods
#

def planckian(nu,T_plk):
    tmp     = (units_h*nu)/(units_kB*T_plk)
    return (2.*units_h*nu**3)/(units_c**2)/(np.exp(tmp)-1.)

def mag_filter(lam,T,rad,dist,ff):
    fnu     = calc_fnu(lam,T,rad,dist,ff)
    fnu     = np.maximum(fnu,np.zeros(fnu.shape)) # avoid unphysical values
    return -2.5*np.log10(fnu)-48.6

def calc_fnu(lam,temp,rad,dist,ff):
    ff1     = ff[:len(ff)//2]
    ff2     = ff[len(ff)//2:]
    tmp1    = rad**2 * ff1 * planckian(units_c/(100.*lam),temp)
    tmp2    = np.flip(rad, axis=1)**2 * ff2 * planckian(units_c/(100.*lam), np.flip(temp, axis=1))
    return (tmp1+tmp2).sum(axis=1)/dist**2

def compute_magnitudes(times,ff,rad_ray,T_ray,lambdas,D,tshift):
    ordered_T       = np.asarray(list(zip(*T_ray)))
    ordered_R       = np.asarray(list(zip(*rad_ray)))
    shifted_times   = times+tshift
    return { li : np.interp(times, shifted_times, mag_filter(lambdas[li],ordered_T,ordered_R,D,ff), left=np.inf) for li in list(lambdas.keys()) }

#
# Spherical expansion methods
#

def mass_gt_v(v,mej,v_exp):
    return mej*units_Msun*(1.0+func_vel(v/v_exp))  #[g]

def func_vel(x):
    return 35.*x**7/112.-105.*x**5/80.+35.*x**3/16.-35.*x/16.

def t_diff_v(kappa,v,m_v,omega):
    return np.sqrt(kappa*m_v/(omega*v*units_c*units_c))  #[s]

def t_fs_v(kappa,v,m_v,omega):
    return np.sqrt(1.5*kappa*m_v/(omega*v*v*units_c*units_c))  #[s]

def GK_expansion_model(args):

    Omega,m_ej,v_rms,v_min,n_v,kappa = args
    v_max  = 3.*v_rms
    vel    = np.linspace(v_min,v_max,n_v)
    m_vel  = mass_gt_v(vel,m_ej,v_max)
    t_diff = t_diff_v(kappa,vel,m_vel,Omega)
    t_fs   = t_fs_v(kappa,vel,m_vel,Omega)
    return vel,m_vel,t_diff,t_fs

#
# Effective temperature
#

def compute_effective_temperature(Lum,dOmega,r_ph):
    return np.power(Lum/(dOmega*r_ph**2*units_sigma_SB), 0.25)
