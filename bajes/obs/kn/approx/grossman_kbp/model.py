from __future__ import division, unicode_literals
import numpy as np

from ..... import SEC_2_DAY
from .utils import compute_magnitudes, GK_expansion_model, compute_effective_temperature

import logging
logger = logging.getLogger(__name__)

# obs. the following universal constant are not expressed in SI
units_c        = 2.99792458e10     #[cm/s]
units_Msun     = 1.98855e33        #[g]
units_sigma_SB = 5.6704e-5         #[erg/cm^2/s/K^4]
units_h        = 6.6260755e-27     #[erg*s]
units_kB       = 1.380658e-16      #[erg/K]
units_pc2cm    = 3.085678e+18      #[cm/pc]

#
# auxiliary methods
#

def initialize_flux_factors(nrays):
    """
        compute flux factors (uniform in the cos-theta)
        from given viewing angle
    """
    import os
    from scipy.interpolate import InterpolatedUnivariateSpline
    from ..... import __path__
    view_angles     = np.linspace(0.0,90.,91)
    data_location   = __path__[0] + '/obs/kn/approx/grossman_kbp/fluxfactors/'
    flux_factors    = np.array([np.loadtxt(data_location+'/ff_ray_%d.dat'%i,unpack=True,usecols=([1])) for i in range(int(nrays))])
    return [InterpolatedUnivariateSpline(view_angles,f) for f in flux_factors]

def initialize_angular_axis(n):
    """
        compute angular distribution (uniform in the cos-theta)
    """
    delta = 1./n
    a = np.array([ [np.arccos(delta*i),np.arccos(delta*(i-1))] for i in range(int(n),0,-1)])
    o = np.array([ 2.*np.pi*(np.cos(x[0]) - np.cos(x[1])) for x in a])
    return a, o

def expand_shell_and_radiate(time, heat, exp_args, eps_args):

    # expand shell ejecta
    vel, m_v, t_diff, t_fs  = GK_expansion_model(exp_args)

    # estimate radiative contribution
    v_fs    = np.interp(time, np.flip(t_fs),   np.flip(vel))
    m_diff  = np.interp(time, np.flip(t_diff), np.flip(m_v))
    m_fs    = np.interp(time, np.flip(t_fs),   np.flip(m_v))
    m_rad   = m_diff-m_fs

    # compute heating rate
    eps     = heat.heating_rate(time, eps_args)

    # compute photosphere radius and bolometric luminosity
    Rph     = v_fs*units_c*time
    Lbol    = m_rad*eps

    return Rph, Lbol

#
# nuclear heating-rate interpolant
#

class Heating(object):

    def __init__(self, interp_kind='linear'):

        from scipy import interpolate

        x = [np.log10(1.e-3),np.log10(5e-3),np.log10(1e-2),np.log10(5e-2)]
        y = [0.1,0.2,0.3]
        a = [[2.01,0.81,0.56,0.27],[4.52,1.90,1.31,0.55],[8.16,3.20,2.19,0.95]]
        b = [[0.28,0.19,0.17,0.10],[0.62,0.28,0.21,0.13],[1.19,0.45,0.31,0.15]]
        d = [[1.12,0.86,0.74,0.60],[1.39,1.21,1.13,0.90],[1.52,1.39,1.32,1.13]]

        # define the interpolation functions
        self.fa = interpolate.interp2d(x, y, a, kind=interp_kind)
        self.fb = interpolate.interp2d(x, y, b, kind=interp_kind)
        self.fd = interpolate.interp2d(x, y, d, kind=interp_kind)

    def heating_rate(self, t, args):
        """
            Compute heating rate according with
            Ref. Korobkin et al. (2012), arXiv:1206.2379 [astro-ph.SR]
        """
        alpha, t0, sigma0, eps_nuc, m_ej, Omega, v_rms = args
        eps_th  = self.therm_efficiency(t=t,m=m_ej,omega=Omega,v=v_rms)
        return eps_nuc*np.power(0.5 - 1./np.pi * np.arctan((t-t0)/sigma0),alpha) * (eps_th/0.5)

    def therm_efficiency_params(self, m, omega, v):
        m_iso = 4.*np.pi/omega * m
        # assign the values of the mass and velocity
        xnew=np.log10(m_iso)   #mass     [Msun]
        ynew=v                 #velocity [c]
        # compute the parameters by linear interpolation in the table
        return self.fa(xnew,ynew),self.fb(xnew,ynew),self.fd(xnew,ynew)

    def therm_efficiency(self,m,omega,v,t):
        """
            Compute thermal efficiency according with
            Barnes et al. (2016), arXiv:1605.07218 [astro-ph.HE]
        """
        a,b,d       = self.therm_efficiency_params(m,omega,v)
        time_days   = t*SEC_2_DAY
        tmp         = 2.*b*np.power(time_days,d)
        return 0.36*(np.exp(-a*time_days) + np.log(1.+tmp)/tmp)

#
# shell object - expand single component
#

class Shell(object):
    """
        Ejecta shell class based on Grossman et al. (2014)
        https://arxiv.org/abs/1307.2943
    """
    def __init__(self, name, geom, time, angles, omegas, heat, v_min, n_v):

        # initilize shell name
        self.name   = name

        if geom not in ['isotropic', 'polar', 'equatorial']:
            logger.error("Unable to set geometry for {} shell. Please use 'isotropic', 'polar' or 'equatorial'.".format(self.name))
            raise ValueError("Unable to set geometry for {} shell. Please use 'isotropic', 'polar' or 'equatorial'.".format(self.name))
        else:
            self.geometry = geom

        # initialize angular axis
        self.time   = time
        self.angles = angles
        self.omegas = omegas

        # initialize heating model
        self.heat   = heat

        # initialize integration constants
        self.v_min  = v_min
        self.n_v    = n_v

    def get_angular_distribution(self, params):
        """
            Return uniform angular profiles for mass, velocity and opacity
            Arguments:
            - params : list of floats, [mej, vel, opac]
                       mej  : total ejected mass
                       vel  : shell rms velocity
                       opac : opacity
            Return:
            - mass profile
            - velocity profile
            - opacity profile
        """
        at = np.transpose(self.angles)

        if self.geometry == 'isotropic':
            m_dist = params['mej_{}'.format(self.name)] * 0.5 * (np.cos(at[0])-np.cos(at[1]))
        elif self.geometry == 'polar':
            m_dist = params['mej_{}'.format(self.name)] * 0.5 * (np.cos(at[0])**3 - np.cos(at[1])**3)
        elif self.geometry == 'equatorial':
            m_dist = params['mej_{}'.format(self.name)] * 0.0625 * (np.cos(3.*at[1]) - 9.*np.cos(at[1]) - np.cos(3.*at[0]) + 9.*np.cos(at[0]))

        v_dist = params['vel_{}'.format(self.name)] * np.ones(12)
        k_dist = params['opac_{}'.format(self.name)] * np.ones(12)
        return m_dist, v_dist, k_dist

    def expansion_angular_distribution(self, params):

        # update angular distributions
        ms,vs,ks = self.get_angular_distribution(params)
        Rs_Ls    = np.array([expand_shell_and_radiate(self.time, self.heat,
                                                      [omega,m_ej,v_rms,self.v_min,self.n_v,kappa],
                                                      [params['eps_alpha'], params['eps_time'], params['eps_sigma'], params['eps0'], m_ej, omega, v_rms])
                             for omega,m_ej,v_rms,kappa in zip(self.omegas,ms,vs,ks)])

        return Rs_Ls[:,0], Rs_Ls[:,1]

#
# multi-component KN object
#

class KorobkinBarnesGrossmanPeregoEtAl(object):
    """
        Lightcurve model based on Perego et al. (2017)
        https://arxiv.org/abs/1711.03982
    """

    def __init__(self, times, lambdas, v_min=1.e-7, n_v=400, t_start=1.):

        # initialize angular axis
        # obs. the inclinations angle is divided in 12 slices
        n_rays = 12
        angles, omegas  = initialize_angular_axis(n_rays//2)
        self.angles = angles
        self.omegas = omegas

        # initialize nuclear heating rate model
        heat    = Heating()

        # check time axis
        if any(times < 0.):
            times += t_start - np.times[0]
        self.times  = times

        # initialize shell components
        self.ncomponents    = 0
        self.components     = []
        logger.warning("Perego-Grossman-Korobkin-et-al. model has been initialized with no components.")

        # initialize quantities
        self.lambdas    = lambdas

        # initialize flux factor interpolator
        self.ff_interp  = initialize_flux_factors(n_rays)

    def compute_lc(self, params):

        # compute Rph e Lbol for every shell
        Rs_Ls   = np.array([ci.expansion_angular_distribution(params) for ci in self.components])
        Rs, Ls  = Rs_Ls[:,0], Rs_Ls[:,1]

        # select the photospheric radius as the maximum between the different photospheric radii
        Rph = np.zeros(np.shape(Rs[0]))
        for i in range(self.ncomponents):
            Rph = np.maximum(Rph,Rs[i])

        # compute the total bolometric luminosity
        Lbol = Ls.sum(axis=0)

        # compute the effective BB temperature based on the photospheric radius and luminosity
        # note: here we would need Tfloor for each component
        Teff = np.asarray([compute_effective_temperature(Lbol[i], oi, Rph[i]) for i, oi in enumerate(self.omegas)])

        return Rph, Lbol, Teff

    def __call__(self, times, params):

        # convert iota and distance
        _iota = params['iota']*180./np.pi
        _dist = params['distance']*1.e6*units_pc2cm

        # compute Rph, Lbol, Teff
        Rph, Lbol, Teff = self.compute_lc(params)

        # compute flux factors
        ff = [fi(_iota) for fi in self.ff_interp]

        return compute_magnitudes(self.times,ff,Rph,Teff,self.lambdas,_dist,params['time_shift'])
