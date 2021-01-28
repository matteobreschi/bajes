from __future__ import division, unicode_literals
import numpy as np

from ... import SEC_2_DAY

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
            Ref. Korobin et al. 2012, arXiv:1206.2379 [astro-ph.SR]
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


