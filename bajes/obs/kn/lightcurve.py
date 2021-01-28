import numpy as np

from .heating import Heating
from .utils import compute_magnitudes, GK_expansion_model, compute_effective_temperature

# obs. the following universal constant are not expressed in SI
units_c        = 2.99792458e10     #[cm/s]
units_Msun     = 1.98855e33        #[g]
units_sigma_SB = 5.6704e-5         #[erg/cm^2/s/K^4]
units_h        = 6.6260755e-27     #[erg*s]
units_kB       = 1.380658e-16      #[erg/K]
units_pc2cm    = 3.085678e+18      #[cm/pc]

#
# initialization methods
#

def initialize_flux_factors(nrays):
    """
        compute flux factors (uniform in the cos-theta)
        from given viewing angle
    """
    import os
    from scipy.interpolate import InterpolatedUnivariateSpline
    from ... import __path__
    
    view_angles     = np.linspace(0.0,90.,91)
    data_location   = __path__[0] + '/obs/kn/fluxfactors/'
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

class Shell(object):
    """
    Ejecta shell class
    """
    def __init__(self, name, time, angles, omegas, heat, v_min, n_v):
        
        # initilize shell name
        self.name   = name
        
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
        m_dist = params['mej_{}'.format(self.name)] * 0.5 * (np.cos(at[0])-np.cos(at[1]))
        v_dist = params['vel_{}'.format(self.name)] * np.ones(12)
        k_dist = params['opac_{}'.format(self.name)] * np.ones(12)
        return m_dist, v_dist, k_dist

    def expansion_angular_distribution(self, params):

        # update angular distributions
        ms,vs,ks = self.get_angular_distribution(params)
        Rph = []
        Lbol = []

        # iterate over angular axis
        for omega,m_ej,v_rms,kappa in zip(self.omegas,ms,vs,ks):

            # expand shell ejecta
            exp_args = [omega,m_ej,v_rms,self.v_min,self.n_v,kappa]
            vel,m_vel,t_diff,t_fs = GK_expansion_model(exp_args)
            v_fs    = np.interp(self.time, t_fs[::-1], vel[::-1])
            m_diff  = np.interp(self.time, t_diff[::-1], m_vel[::-1])
            m_fs    = np.interp(self.time, t_fs[::-1], m_vel[::-1])
            m_rad   = m_diff-m_fs

            # compute photosphere radius
            Rph.append(v_fs*units_c*self.time)
            
            # compute bolometric luminosity
            eps_args = [params['eps_alpha'], params['eps_time'], params['eps_sigma'], params['eps0'], m_ej, omega, v_rms]
            eps      = self.heat.heating_rate(self.time, eps_args)
            Lbol.append(m_rad*eps)

        return np.array(Rph), np.array(Lbol)

class Lightcurve(object):

    def __init__(self, comps, times, lambdas,
                 v_min=1.e-7, n_v=400, t_start=1.):
        
        # initialize angular axis
        # obs. the inclinations angle is divided in 12 slices
        n_rays = 12
        angles, omegas  = initialize_angular_axis(n_rays//2)
            
        # initialize nuclear heating rate model
        heat    = Heating()
        
        # check time axis
        if any(times < 0.):
            times += t_start - np.times[0]
        self.times  = times
        
        # initialize shell components
        self.ncomponents    = len(comps)
        self.components     = [Shell(ci, times, angles, omegas, heat, v_min, n_v) for ci in comps]
        
        # initialize filter bands
        self.lambdas    = lambdas
        
        # initialize flux factor interpolator
        self.ff_interp  = initialize_flux_factors(n_rays)
    
    def compute_lc(self, params):

        # compute Rph e Lbol for every shell
        Rs = []
        Ls = []
        for ci in self.components:
            ri, li = ci.expansion_angular_distribution(params)
            Rs.append(ri)
            Ls.append(li)
        
        # select the photospheric radius as the maximum between the different single photospheric radii
        Rph = np.zeros(np.shape(ri))
        for i in range(self.ncomponents):
            Rph = np.maximum(Rph,Rs[i])

        # compute the total bolometric luminosity
        Lbol = np.sum(Ls, axis = 0)

        # compute the effective BB temperature based on the photospheric radius and luminosity
        Teff = [[compute_effective_temperature(L,ci.omegas[k],R) for L,R in zip(Lbol[k,:],Rph[k,:])] for k in range(len(ci.angles))]

        return np.array(Rph), np.array(Lbol), np.asarray(Teff)

    def compute_mag(self, params):
        
        if 'cosi' in params.keys():
            params['iota'] = np.arccos(params['cosi'])
        elif 'iota' in params.keys():
            params['cosi'] = np.cos(params['iota'])
        else:
            raise KeyError("Unable to read inclination parameter, information is missing.\n Please use iota or cosi.")
        
        # convert iota and distance
        _iota = params['iota']*360./(2.*np.pi)
        _dist = params['distance']*1.e6*units_pc2cm
        
        # compute Rph, Lbol, Teff
        Rph, Lbol, Teff = self.compute_lc(params)

        # compute flux factors
        ff = [fi(_iota) for fi in self.ff_interp]

        return compute_magnitudes(self.times,ff,Rph,Teff,self.lambdas,_dist,params['time_shift'])








