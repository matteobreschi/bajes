from __future__ import division, unicode_literals, absolute_import
import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d

import logging
logger = logging.getLogger(__name__)

from ... import MRSUN_SI

class cgs:
    c           = 2.99792458e10
    c2          = c*c
    G           = 6.6730831e-8
    Msun        = 1.988435e33
    km_2_Msun   = G/c2

#EoS class using dense eos from Read et al (2009)
def get_eos_from_spectral_params(ll, trans = None):

    # ll = [g0, g1, g2, g3]

    #dense eos starting pressure
    p1 = 10**ll[0]

    #polytrope indices
    g1 = ll[1]
    g2 = ll[2]
    g3 = ll[3]

    #transition densities
    if trans is None:
        r1 = 2.8e14
        r2 = 10**14.7
        r3 = 1e15
    else:
        r1 , r2, r3 = trans

    #scaling constants
    K1 = p1/(r2**g1)
    K2 = K1 * r2**(g1-g2)
    K3 = K2 * r3**(g2-g3)

    tropes  = [Monotrope(K1, g1), Monotrope(K2, g2), Monotrope(K3, g3)]
    trans   = [r1, r2, r3]
    return Polytrope(tropes, trans)

# Smoothly glue core to SLy crust
# for polytropic eos we can just unpack
# and repack the piecewise presentation
def glue_crust_and_core(crust, core):

    #unpack crust and core
    tropes_crust = crust.tropes
    trans_crust  = crust.transitions

    tropes_core = core.tropes
    trans_core  = core.transitions

    #find transition depth
    rho_tr = (tropes_core[0].K / tropes_crust[-1].K )**( 1.0/( tropes_crust[-1].G - tropes_core[0].G ) )
    trans_core[0] = rho_tr

    #repack
    tropes = tropes_crust + tropes_core
    trans  = trans_crust  + trans_core

    for trope in tropes:
        trope.a = 0.0

    return Polytrope( tropes, trans )

def get_SLy_crust_eos():
    ##################################################
    #SLy (Skyrme) crust
    KSLy = [6.80110e-9, 1.06186e-6, 5.32697e1, 3.99874e-8]  #Scaling constants
    GSLy = [1.58425, 1.28733, 0.62223, 1.35692]             #polytropic indices
    RSLy = [1.e4, 2.44034e7, 3.78358e11, 2.62780e12 ]       #transition depths

    tropes = []
    trans = []

    pm = None
    for (K, G, r) in zip(KSLy, GSLy, RSLy):
        m = Monotrope(K*cgs.c**2, G)
        tropes.append( m )

        #correct transition depths to avoid jumps
        rho_tr = r
        trans.append(rho_tr)

    #Create crust using polytrope class
    return Polytrope(tropes, trans)

#Monotropic eos
class Monotrope(object):

    #transition continuity constant
    a = 0.0

    def __init__(self, K, G):
        self.K = K / cgs.c2
        self.G = G
        self.n = 1.0/(G - 1)

    #pressure P(rho)
    def pressure(self, rho):
        return cgs.c2 * self.K * rho**self.G

    #energy density mu(rho)
    def edens(self, rho):
        return (1.0 + self.a)*rho + (self.K/(self.G - 1)) * rho**self.G

    #energy density derivative de/dP(P)
    def dedp(self, press):
        _scale = (cgs.c2 * self.K)**(1./self.G)
        return (1.0 + self.a)*(press**(1./self.G -1.))/_scale/self.G + 1./((self.G - 1)*cgs.c2)

    #for inverse functions lets define rho(P)
    def rho(self, press):
        if press < 0.0:
            return 0.0
        return ( press/cgs.c2/self.K )**(1. / self.G)

# Piecewise polytropic EOS
class Polytrope(object):

    def __init__(self, tropes, trans, prev_trope = None ):
        self.tropes      = tropes
        self.transitions = trans

        prs  = []
        eds  = []

        for (trope, transition) in zip(self.tropes, self.transitions):

            if not( prev_trope == None ):
                trope.a = self._ai( prev_trope, trope, transition )
            else:
                transition = 0.0

            ed = trope.edens(transition)
            pr = trope.pressure(transition)

            prs.append( pr )
            eds.append( ed )

            prev_ed = ed
            prev_tr = transition
            prev_trope = trope

        self.prs = np.array(prs)
        self.eds = np.array(eds)

    def _is_physical(self, pmax):

        # monotonicity
        de = np.diff(self.eds)
        dp = np.diff(self.prs)
        e_err = len(np.where(de < 0)[0])
        p_err = len(np.where(dp < 0)[0])
        if p_err or e_err:
            mono_flag = False
        else:
            mono_flag = True

        # causality
        quasi_vmax = self.dedp(pmax)
        if quasi_vmax < 0 or (not np.isfinite(quasi_vmax)):
            cause_flag = False
        else:
            vmax = 1./np.sqrt(quasi_vmax)
            if vmax > cgs.c:
                cause_flag = False
            else:
                cause_flag = True

        return (mono_flag and cause_flag)

    def _ai(self, pm, m, tr):
        return pm.a + (pm.K/(pm.G - 1))*tr**(pm.G-1) - (m.K/(m.G - 1))*tr**(m.G-1)

    def _find_interval_given_density(self, rho):
        if rho <= self.transitions[0]:
            return self.tropes[0]

        i = np.concatenate(np.where((self.transitions[:-1] <= rho)&(rho < self.transitions[1:])))
        if len(i) > 0:
            return self.tropes[np.min(i)]
        else:
            return self.tropes[-1]

    #inverted equations as a function of pressure
    def _find_interval_given_pressure(self, press):

        if press <= self.prs[0]:
            return self.tropes[0]

        i = np.concatenate(np.where((self.prs[:-1] <= press)&(press < self.prs[1:])))
        if len(i) > 0:
            return self.tropes[np.min(i)]
        else:
            return self.tropes[-1]

    ##################################################
    def pressure(self, rho):
        trope = self._find_interval_given_density(rho)
        return trope.pressure(rho)

    def pressures(self, rhos):
        return np.array([self.pressure(ri) for ri in rhos])

    def dedp(self, press):
        trope = self._find_interval_given_pressure(press)
        return trope.dedp(press)

    def edens_inv(self, press):
        trope = self._find_interval_given_pressure(press)
        rho = trope.rho(press)
        trope = self._find_interval_given_density(rho)
        return trope.edens(rho)

    def rho(self, press):
        trope = self._find_interval_given_pressure(press)
        return trope.rho(press)

class TOVSolver(object):
    """
        TOV solver object for parametrized spectral EOS (with 4 coefficients)
    """

    def __init__(self, gammas=None, transitions=None,
                 polytrope = None,
                 n=750, rtol=1.e-8, atol=1.e-8, rmin=1., rmax=2e6,
                 logrhomin=14., logrhomax=16.):

        # initialize EOS
        if isinstance(polytrope, Polytrope):
            self.EOS    = polytrope
        else:
            if gammas is None:
                logger.error("Unable to initialize TOVSolver, please provide gamma parameters.")
                raise RuntimeWarning("Unable to initialize TOVSolver, please provide gamma parameters.")
            core_eos    = get_eos_from_spectral_params(gammas, trans=transitions)
            crust_eos   = get_SLy_crust_eos()
            self.EOS    = glue_crust_and_core(crust_eos, core_eos)

        # initialize ODE integrator parameters
        self.N      = int(n)
        self.rtol   = rtol
        self.atol   = atol
        self.rmin   = rmin
        self.rmax   = rmax
        self.rhomin = logrhomin
        self.rhomax = logrhomax

        # solve M-R diagramm
        self.mass, self.radius, self.rhoc = self.mass_radius()
        self.Mmax       = np.max(self.mass)
        self.Rmax       = self.radius[np.argmax(self.mass)]
        self.rhocmax    = self.rhoc[np.argmax(self.mass)]

        # remove unphysical branch
        indxs = np.where(self.radius >= self.Rmax)
        self.mass, self.radius, self.rhoc = self.mass[indxs], self.radius[indxs], self.rhoc[indxs]

        # check is physical
        Pmax = self.EOS.pressure(self.rhoc[np.argmax(self.mass)])
        if not self.EOS._is_physical(Pmax):
            self.is_physical = False
            #logger.warning("Requested parameters [{:.1f},{:.2f},{:.2f},{:.2f}] gave an unphysical EOS.".format(gammas[0], gammas[1], gammas[2], gammas[3]))
        else:
            self.is_physical = True

    def tov(self, y, r):

        P, m = y
        eden = self.EOS.edens_inv( P )

        dPdr = -cgs.G*(eden + P/cgs.c2)*(m + 4.0*np.pi*r**3*P/cgs.c2)
        dPdr = dPdr/(r*(r - 2.0*cgs.km_2_Msun*m))
        dmdr = 4.0*np.pi*r**2*eden

        return [dPdr, dmdr]

    def tovsolve(self, rhoc):

        r = np.linspace(self.rmin, self.rmax, self.N)
        P = self.EOS.pressure( rhoc )
        eden = self.EOS.edens_inv( P )
        m = 4.0*np.pi*r[0]**3*eden

        psol, _ = odeint(self.tov, [P, m], r, rtol=self.rtol, atol=self.atol, full_output=1)
        return r, psol[:,0], psol[:,1]

    def mass_radius(self):

        N       = self.N//4
        mcurve  = np.zeros(N)
        rcurve  = np.zeros(N)
        rhocs   = np.logspace(self.rhomin, self.rhomax, N)

        j   = 0
        dm  = 1

        while dm > 0:

            rad, press, mass = self.tovsolve(rhocs[j])
            rad  /= 1.0e5 #cm to km
            mass /= cgs.Msun

            indx        = np.max(np.where(press>0))
            mcurve[j]   = mass[indx]
            rcurve[j]   = rad[indx]

            dm = mcurve[j] - mcurve[j-1]

            j += 1
            if j == len(rhocs):
                break

        return mcurve[:j], rcurve[:j], rhocs[:j]

    def love(self, param, r, R_dep):
        beta, H = param
        e_R, p_R, m_R = R_dep

        p = p_R(r)
        de_dp = self.EOS.dedp(p)

        dbetadr = H * (-2 * np.pi * cgs.G / cgs.c2 * ( 5 * e_R(r) + 9 * p_R(r) /cgs.c2 + de_dp * cgs.c2 * (e_R(r) + p_R(r) / cgs.c2)) + 3 / r ** 2 + 2 * (1 - 2 * m_R(r) / r * cgs.km_2_Msun) ** (-1) * ( m_R(r) / r ** 2 * cgs.km_2_Msun + cgs.G / cgs.c2 ** 2 * 4 * np.pi * r * p_R(r)) ** 2) + beta / r * (-1 + m_R(r) / r * cgs.km_2_Msun + 2 * np.pi * r ** 2 * cgs.G / cgs.c2 * (e_R(r) - p_R(r) / cgs.c2))
        dbetadr *= 2 * (1 - 2 * m_R(r) / r * cgs.km_2_Msun) ** (-1)

        dHdr = beta
        return [dbetadr, dHdr]

    def love_number(self, M):

        # if M > self.Mmax: DO SOMETHING

        # estimate central density and radius
        rhoc    = np.interp(M, self.mass, self.rhoc)
        R       = np.interp(M, self.mass, self.radius)

        R *= 1e5
        M *= cgs.Msun

        # get property profiles
        r, p, m = self.tovsolve(rhoc)

        e   = np.array([self.EOS.edens_inv(pi) for pi in p])
        e_R = interp1d(r, e, kind='linear', bounds_error=False, fill_value="extrapolate")
        p_R = interp1d(r, p, kind='linear', bounds_error=False, fill_value="extrapolate")
        m_R = interp1d(r, m, kind='linear', bounds_error=False, fill_value="extrapolate")

        beta0 = 2 * r[0]
        H0 = r[0] ** 2

        # solve Love ODE
        solution = odeint(self.love, [beta0, H0], r, args=([e_R, p_R, m_R],), rtol=self.rtol)
        beta = solution[-1, 0]
        H = solution[-1, 1]
        y = R * beta / H
        C = M / R * cgs.km_2_Msun
        k2 = 8. / 5. * C ** 5. * (1. - 2. * C) ** 2. * (2. + 2. * C * (y - 1.) - y) * (2. * C * (6. - 3. * y + 3. * C * (5. * y - 8.)) + 4. * C ** 3. * (13. - 11. * y + C * (3. * y - 2.) + 2. * C ** 2. * (1 + y)) + 3. * (1. - 2. * C) ** 2. * (2. - y + 2. * C * (y - 1.)) * (np.log(1. - 2. * C))) ** (-1)
        return k2, R/1e5

    def tidal_deformability(self, M):

        # check mass, otherwise return BH
        if M > self.Mmax:
            return 0.
        else:
            k2, R = self.love_number(M)
            R_M   = R*1e3/(M*MRSUN_SI)
            return (2./3.)*k2*np.power(R_M,5.)
