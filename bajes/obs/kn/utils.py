import numpy as np

# obs. the following universal constant are not expressed in SI
units_c        = 2.99792458e10     #[cm/s]
units_Msun     = 1.98855e33        #[g]
units_sigma_SB = 5.6704e-5         #[erg/cm^2/s/K^4]
units_h        = 6.6260755e-27     #[erg*s]
units_kB       = 1.380658e-16      #[erg/K]
units_pc2cm    = 3.085678e+18      #[cm/pc]

#
# NR-informed relations
#

def NRfit_recal_mass_dyn(mchirp, q, lambda1, lambda2, NR_fit_recal_mdyn, **kwargs):
    mtot        = mchirp / (q/(1+q)**2)**0.6
    log_mdyn    = NRfit_log_mass_dyn(mtot, q, lambda1, lambda2) * (1. + NR_fit_recal_mdyn)
    mdyn        = mtot * np.exp(log_mdyn)
    return np.max([0., mdyn])

def NRfit_recal_vel_dyn(mchirp, q, lambda1, lambda2, NR_fit_recal_vdyn, **kwargs):
    mtot    = mchirp / (q/(1+q)**2)**0.6
    vdyn    = NRfit_vel_dyn(mtot, q, lambda1, lambda2) * (1. + NR_fit_recal_vdyn)
    return np.max([1e-10, vdyn])

def NRfit_recal_mass_wind(mchirp, q, lambda1, lambda2, disk_frac, **kwargs):
    mtot        = mchirp / (q/(1+q)**2)**0.6
    log_m_disk  = NRfit_log_mass_disk(mtot, q, lambda1, lambda2)
    mwind       = mtot * np.exp(log_m_disk) * disk_frac
    return np.max([0., mwind])

def NRfit_log_mass_dyn(mtot, q, lambda1, lambda2):
    """
        NR-calibrated relation for mass of dynamical ejecta
        Returns log(m_ej/M) where M = m1 + m2 (natural log)
    """
    a0, n0, b1, b2, c1, c2 = [-21.295092178221847, 1.9743123050205846,
                              0.0044685525694660435, -0.0024627659648901452,
                              -0.5258273201873834, -0.23928655421412218]

    nu     = q/(1+q)**2
    m1     = mtot * q / (1+q)
    m2     = mtot / (1+q)
    corr_q = 1. + n0*(1.-4.*nu)
    corr_p = 1. + b1*np.sqrt(lambda1) + b2*np.sqrt(lambda2) + c1*m1**(-0.25) + c2*m2**(-0.25)
    return a0 * corr_q * corr_p

def NRfit_vel_dyn(mtot, q, lambda1, lambda2):
    """
        NR-calibrated relation for velocity of dynamical ejecta
        Returns v_ej / c
    """
    a0, n0, b1, b2, c1, c2 = [0.09217372, -4.52017477, -0.02171866, 0.00946049, -0.2176058, 1.32125944]

    nu     = q/(1+q)**2
    m1     = mtot * q / (1+q)
    m2     = mtot / (1+q)
    corr_q = 1. + n0*(1.-4.*nu)
    corr_p = 1. + b1*np.sqrt(lambda1) + b2*np.sqrt(lambda2) + c1*m1 + c2*m2
    return a0 * corr_q * corr_p

def NRfit_log_mass_disk(mtot, q, lambda1, lambda2):
    """
        NR-calibrated relation for disk mass
        Returns log(m_disk/M) where M = m1 + m2 (natural log)
    """
    alpha, a1, a2, b1, b2, Lbar, Sbar, Abar = [-13.846080565670077,
                                                4.977942316040381e-06, 1.8902832916214914e-06,
                                                -0.4708068240433623, 0.33378530243025306,
                                                558.1230475510761, -176.21011658144016, 1.0095398304221503]

    m1     = mtot * q / (1+q)
    m2     = mtot / (1+q)
    corr_l = 1. + Abar*((1./np.pi)*np.arctan((lambda1+lambda2-Lbar)/Sbar) - 0.5)
    corr_p = 1 + a1*(lambda1)**2 + a2*(lambda2)**2 + b1*m1**2 + b2*m2**2
    return alpha * corr_l * corr_p
