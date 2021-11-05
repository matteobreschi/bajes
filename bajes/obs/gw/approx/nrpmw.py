from __future__ import division, unicode_literals, absolute_import
import numpy as np
from scipy.special import erfcx, erfi, factorial, binom

from .... import MTSUN_SI, MRSUN_SI, PARSEC_SI

import logging
logger = logging.getLogger(__name__)

PIHALF      = np.pi/2.
TWOPI       = 2.*np.pi
SQRTPI      = np.sqrt(np.pi)
EmIPIHALF   = np.exp(-1j*PIHALF)

SPHARM_22   = np.sqrt(5./(4.*np.pi))
WI_FACT     = -0.5j*SQRTPI
_prefact    = SPHARM_22*MRSUN_SI*MTSUN_SI/1e6/PARSEC_SI

from ..utils import lambda_2_kappa

#####################################################
#                                                   #
#           List of empirical relations             #
#                                                   #
#####################################################

__CS__ = {  'f_2':      (0.0881,22.814700323004796,0.29246401361707447,24.999999962619935,
                         0.007023135582189979,-1.7816176582640582e-06,0.025874168850673,6.579585900552341e-06,
                         5.427803269576005,0.0,39.29307296443815,0.0),
            'f_0':      (0.02734087474437463,19.31605961171479,-1.8572238969702572,-75.76657196173291,
                         -0.002967055067321701,8.483851129151385e-06,0.008584328745393421,0.0,
                         20.485528708094723,21.495961009283235,10.469797211012333,0.0),
            'f_m':      (0.22754806, 0.92330138, 0.59374838, -1.99378496,
                         0.03445340731627873, 5.5799962023491245e-06, 0.08404652974611324, 0.00011328992320789428,
                         13.828175998146255, 517.4149218303298, 12.74661916436829, 139.76057108600236),
            'a_m':      (0.39475762, -1.13292325, -0.02991597, -2.59318042,
                         0.03901957425837708, 5.184564561045753e-05, 0.06032721528620493, 0.00013795694839938442,
                         10.410256292591564,54.513466134598985,10.826296683028199,54.53588176973234),
            '1_t_0':    (0.03265128,0.29942023,-0.23287941,4.76753546,
                         0.003583706549541881,0.0,0.010534338544905889,0.0,
                         -11.957632886058573,0.0,-3.219650560016596,0.0),
            'a_0':      (0.02356281923775041,0,1.0770066927080064,260.4081790088421,
                         -0.0013182467077588536,0.0,0.0,0.0,-4.31438985,0.0,0.0,0.0),
            'a_1':      (-0.0029226779553209233,-6.1874352287025935,-1.0605928759169487,101.44196137082086,
                         -6.706323210333894,0.003979772810911221,0.09585169896475423,0.0,
                         -5.421270624526479,0.0,2.8144039565646857,0.0),
            'a_2':      (0.1666825236292765,-5.134506309621276,-3.7958637378393543,-28.468793005957988,
                         0.0,0.0,0.005774488759045594,0.0,0.0,0.0,4.0272943907508034e-08,0.0),
            'a_3':      (0.1661508508278642,0.10715850717230395,-2.045622790310528,-45.056062822266554,
                         -7.060079453855475e-05,0.0,0.0013540341811212075,0.0,
                         -1423.3380498964327,0.0,284.69248124488655,0.0),
            'df_m':     (0.00744217,-1.79881866,0.35550959,-7.16735727,
                         0.013899742515873507,-2.424960263371662e-05,0.05883319329248303,0.0001881995928594438,
                         -28.636629921974894,-36.18381939817666,19.529300938536355,7.08859053788995),
            'B_2':      (0.1911683981413522,4.07380693412795,-1.5731131167054548,99.99999646119788,
                         0.058836882420503916,0.0,3.8959139356418375,0.0,-5.293429864052918,0.0,0.0,0.0),
            'D_2':      (0.051391957484555766,0.49441169605850593,-3.73435517876452,-145.02260655057313,
                         -0.006250243266645933,1.7275049934470065e-05,0.019436925023348747,0.0,
                         -7.935827885997765,1.8819401548389918,99.99997085474375,0.0),
            'G_2':      (0.16366306015470958,209.2791565519091,-0.2996818803038818,24.999997828247693,
                         0.021947463599187318, 0.0, 0.3527862769626609, 0.0,
                         -0.5110803184473908,0.0,74.7195642043283,0.0),
}

__ERRS__ = {    'f_2':      0.03925186372160585,
                'f_0':      0.4492392741185888,
                'f_m':      0.026091601334595742,
                'a_m':      0.017658369779394143,
                '1_t_0':    0.09170663395938666,
                'a_0':      0.6634245499270375,
                'a_1':      0.2762013149125831,
                'a_2':      0.3848413412777949,
                'a_3':      0.26938411798665807,
                'df_m':     0.7508899847722584,
                'B_2':      0.2702931305277849,
                'D_2':      0.7437024292273079,
                'G_2':      0.9770102502362071
}

__BNDS__ = {    'f_2':      [-0.5,0.5],
                'f_0':      [-1.,2.],
                'f_m':      [-0.5,0.5],
                'a_m':      [-0.5,0.5],
                '1_t_0':    [-0.5,0.5],
                'a_0':      [-1.,4.],
                'a_1':      [-1.,2.],
                'a_2':      [-1.,2.],
                'a_3':      [-1.,2.],
                'df_m':     [-4.,4.],
                'B_2':      [-1.,2.],
                'D_2':      [-1.,4.],
                'G_2':      [-1.,4.]
}

__recalib_names__ = list(__CS__.keys())

def __fit_func__(pars, key):
    # get coefficients and params
    a0, a1, b0, b1, n1, n2, d1, d2, q1, q2, q3, q4 = __CS__[key]
    kkk,sss,qqq = pars['k2t'], pars['Shat'], pars['X']
    # compute fit
    p1s = b0 * (1. + b1 * qqq)
    _n1 = n1 * (1. + q1 * qqq)
    _n2 = n2 * (1. + q2 * qqq)
    _d1 = d1 * (1. + q3 * qqq)
    _d2 = d2 * (1. + q4 * qqq)
    _up = (1.+ _n1 * kkk + _n2 * kkk **2.)
    _lo = (1.+ _d1 * kkk + _d2 * kkk **2.)
    return a0 * (1. + a1*qqq) * (1. + p1s*sss) *  _up / _lo

def _nrpmw_fits(pars, recalib=False):
    """
        Compute PM empirical relations given pars (dict)
    """
    # set variables
    m1              = pars['mtot']/(1. + 1./pars['q'])
    m2              = pars['mtot']/(1. + pars['q'])
    pars['nu']      = m1*m2/(pars['mtot']**2.)
    pars['X']       = 1. - 4.*pars['nu']
    pars['Shat']    = (pars['s1z'] * pars['q']**2 + pars['s2z']) / (1. + pars['q'])**2
    pars['k2t']     = lambda_2_kappa(m1, m2, pars['lambda1'], pars['lambda2'])
    # compute fits
    if recalib:
        fits = {ki : __fit_func__(pars, ki)*(1.+pars['NRPMw_recal_'+ki]) for ki in __recalib_names__}
    else:
        fits = {ki : __fit_func__(pars, ki) for ki in __recalib_names__}
    # join dicts
    pars = {**pars, **fits}
    # check amplitudes
    pars['a_0']  = max(1e-12,pars['a_0'])
    pars['a_1']  = max(1e-12,pars['a_1'])
    pars['a_2']  = max(0.,pars['a_2'])
    pars['a_3']  = max(1e-12,pars['a_3'])
    # check freqs
    pars['f_m']  = pars['f_m'] * pars['nu']
    pars['df_m'] = pars['df_m'] * pars['nu']
    pars['B_2']  = max(0.,pars['B_2'])
    pars['D_2']  = max(0.,pars['D_2'])
    # check times
    pars['t_0']             = 1./pars['1_t_0']
    dt0                     = 0.5/pars['f_0']
    pars['NRPMw_t_coll']    = min(pars['NRPMw_t_coll'], pars['seglen']/(pars['mtot']*MTSUN_SI) - pars['t_0'])
    pars['t_1']             = min(pars['NRPMw_t_coll'], pars['t_0'] + dt0)
    pars['t_2']             = min(pars['NRPMw_t_coll'], pars['t_0'] + 2.*dt0)
    pars['t_3']             = min(pars['NRPMw_t_coll'], pars['t_0'] + 3.*dt0)
    return pars

#####################################################
#                                                   #
#           Baseline wavelet functions              #
#                                                   #
#####################################################

def _wavelet_integral_extremes_erfcx(x,b):
    """
        Compute Gaussian integral given the extremal values
        Gaussian case, generic
    """
    ### compute erfcx integral
    xi  = 1j * x
    return -WI_FACT*( np.exp( b*(b-2j*xi) + np.log(erfcx(xi + 1j*b), dtype=complex) ) - erfcx(xi) )

def _gamma_func(m,x):
    """
        Compute Gamma function using polynomial approximation,
        valid only for m > 0 (integer)
    """
    if m==1:
        return np.exp(-x)
    else:
        return np.exp(-x)*factorial(m-1)*sum([x**k/factorial(k) for k in range(0,m)])

def _integral_xn_exp(n, alpha, tau, ax):
    """
        Compute n-th order approximation of approximated wavelet
        corresponding to the integral of t^n exp(ax * t) dt from 0 to tau
    """
    m = 2*n
    return (alpha**n/factorial(n)) * (_gamma_func(m+1, -ax*tau) - factorial(m)) / ax**(m+1)

def _wavelet_func_exponential(freq, beta, eta, tau):
    """
        Compute Gaussian wavelet
        Exponential case, alpha = 0
    """
    x1      = beta-1j*TWOPI*freq
    c1      = 0.5*eta*(np.exp(x1*tau) - 1. )/x1
    beta    = np.conj(beta)
    eta     = np.conj(eta)
    x1      = beta-1j*TWOPI*freq
    c2      = 0.5*eta*(np.exp(x1*tau) - 1. )/x1
    return  c1 + c2

def _wavelet_func_smallalpha(freq, alpha, beta, eta, tau):
    """
        Compute Gaussian wavelet
        Gaussian case, alpha < 1
    """
    nmax    = 4
    x1      = beta-1j*TWOPI*freq
    cn1     = sum([_integral_xn_exp(ni, alpha, tau, x1) for ni in range(nmax+1)])
    alpha   = np.conj(alpha)
    beta    = np.conj(beta)
    eta     = np.conj(eta)
    x1      = beta-1j*TWOPI*freq
    cn2     = sum([_integral_xn_exp(ni, alpha, tau, x1) for ni in range(nmax+1)])
    return 0.5*eta*(cn1 + cn2)

def _wavelet_func_generic(freq, alpha, beta, eta, tau):
    """
        Compute Gaussian wavelet
        Gaussian case, generic
    """
    sqrta   = np.sqrt(alpha,dtype=complex)
    x1      = 0.5*(beta-1j*TWOPI*freq)/sqrta
    b1      = sqrta*tau
    c1      = 0.5*eta/sqrta
    alpha   = np.conj(alpha)
    beta    = np.conj(beta)
    eta     = np.conj(eta)
    sqrta   = np.sqrt(alpha,dtype=complex)
    x2      = 0.5*(beta-1j*TWOPI*freq)/sqrta
    b2      = sqrta*tau
    c2      = 0.5*eta/sqrta
    out     = c1 * _wavelet_integral_extremes_erfcx(x1,b1) + c2 * _wavelet_integral_extremes_erfcx(x2,b2)

    return out

def _wavelet_func(freq, eta, alpha, beta, tau, tshift=0):
    """
        Frequency-domain representation of

            W(t) = eta * exp( alpha * t**2 + beta*t )

        with eta, alpha, beta in C integrated over t in [0, tau]
        and shifted of a time tshift

        Arguments:
        - freq              : frequency axis
        - eta               : scale factor, complex
        - alpha             : acceleration parameter, complex
            - real(alpha)   : inverse-opposite amplitude width
            - imag(alpha)   : frequency slope
        - beta              : frequency parameter, complex
            - real(beta)    : inverse-opposite damping time
            - imag(beta)    : frequency
        - tau               : strain duration
        - tshift            : additional time-shift, float
    """
    # null wavelet if tau is negative
    if tau <= 0: return 0.

    eta     = complex(eta)
    alpha   = complex(alpha)
    beta    = complex(beta)

    ab_alpha_tau2 = np.abs(alpha)*tau**2
    re_alpha_tau2 = np.abs(np.real(alpha))*tau**2

    if ab_alpha_tau2 < 1e-12:
        # exponential case, alpha = 0
        model = _wavelet_func_exponential(freq, beta, eta, tau)

    elif ab_alpha_tau2 < 1. :
        # gaussian case, alpha < 1
        model   = _wavelet_func_smallalpha(freq, alpha, beta, eta, tau)
    else:
        # check R(a)*t^2 in order to avoid overflows
        if re_alpha_tau2 > 500.:
            logger.debug("Setting upper bound on wavelet parameters in order to avoid numerical overflow.")
            tau = np.sqrt(500./np.abs(np.real(alpha)))
        # gaussian case, general
        model = _wavelet_func_generic(freq, alpha, beta, eta, tau)

    return model * np.exp(-1j*TWOPI*freq*tshift)

def _fm_wavelet_func(freq, eta, alpha, beta, tau, tshift, Omega, Delta, Gamma, Phi):
    """
        Frequency-domain representation of frequency-modulated wavelet;

            W(t) = eta * exp( alpha * t**2 + beta*t ) * exp( -i F(t) )

        with eta, alpha, beta in C integrated over t in [0, tau]
        and shifted of a time tshift.
        F(t) is a damped sinusoidal function that defines the frequency modulations (FMs)
        and it is characterized by the frequency Omega, inverse damping time Gamma,
        initial phase Phi and amplitude Delta

        Arguments:
        - freq              : frequency axis
        - eta               : scale factor, complex
        - alpha             : acceleration parameter, complex
            - real(alpha)   : inverse-opposite of amplitude width [<= 0]
            - imag(alpha)   : frequency slope
        - beta              : frequency parameter, complex
            - real(beta)    : inverse-opposite of damping time [<= 0]
            - imag(beta)    : frequency
        - tau               : strain duration
        - tshift            : additional time-shift, float
        - Omega             : FM frequency
        - Delta             : FM initial amplitude
        - Gamma             : FM inverse damping time
        - Phi               : FM initial phase
    """

    # compute baseline
    h0  = _wavelet_func(freq, eta, alpha, beta, tau, tshift)

    if np.abs(Delta) > 1e-12 and tau > 0:

        # set constants
        nu          = -Gamma - 1j*Omega
        d           = -0.5*Delta/np.abs(nu)**2
        phi_extra   = np.exp(1j*2.*d*(Gamma*np.sin(Phi)+Omega*np.cos(Phi)))
        nmax        = min(max(1,int(2.*(1.+np.abs(Delta/Omega)))),8)

        # compute FM corrections
        h0  = (h0 + sum([(d**n/factorial(n))*sum([binom(n,k)*((-np.conj(nu))**k)*((nu)**(n-k))*_wavelet_func(freq,
                                                                                                             eta*np.exp(-1j*Phi*(n-2*k)),
                                                                                                             alpha,
                                                                                                             beta-n*Gamma-1j*Omega*(n-2*k),
                                                                                                             tau, tshift)
                                                        for k in range(n+1)])
                            for n in range(1,nmax+1)]))*phi_extra

    return h0

#####################################################
#                                                   #
#                   NRPMw model                     #
#                                                   #
#####################################################

def NRPMw(freqs, params, recalib=False):
    """
        Compute NRPMw model given frequency axis (np.array) and parameters (dict)
    """

    freqs  = freqs*MTSUN_SI*params['mtot']
    params = _nrpmw_fits(params, recalib=recalib)

    # initialize complex array
    h22 = np.zeros(len(freqs), dtype=complex)

    # h_fusion,
    # early post-merger corresponding to the fusion of the NS cores
    h22 = h22 + _wavelet_func(freqs,
                              eta     = params['a_m'],
                              alpha   = np.log(params['a_0']/params['a_m'])/(params['t_0']**2)-1j*np.pi*params['df_m'],
                              beta    = -1j*TWOPI*params['f_m'],
                              tau     = params['t_0'])

    # if lambda1 or lambda2 = 0 , avoid PM segment
    if params['lambda1'] < 1 or params['lambda2'] < 1:
        h22 *= _prefact*(params['mtot']**2./params['distance'])*np.exp(-1j*params['phi_ref'])
        return h22*(0.5*(1.+params['cosi']**2.)), h22*(params['cosi']*EmIPIHALF)

    if params['NRPMw_t_coll'] > params['t_0'] :

        # h_recoil,
        # bounce-back of the remnant after the quasi-spherical node
        phi_bounce = params['NRPMw_phi_pm'] + TWOPI*params['f_m']*params['t_0'] + np.pi*params['df_m']*params['t_0']**2.
        h22 = h22 + _fm_wavelet_func(freqs,
                                     eta     = params['a_0']*np.exp(-1j*phi_bounce),
                                     alpha   = np.log(params['a_0']/params['a_1'])/(params['t_1']-params['t_0'])**2 ,
                                     beta    = 2*np.log(params['a_1']/params['a_0'])/(params['t_1']-params['t_0']) - 1j*TWOPI*params['f_2'],
                                     tau     = params['t_1']-params['t_0'],
                                     tshift  = params['t_0'],
                                     Omega   = TWOPI*params['f_0'],
                                     Delta   = params['D_2'],
                                     Gamma   = 0.,
                                     Phi     = -PIHALF)

    if params['NRPMw_t_coll'] > params['t_1'] :

        # h_pulse,
        # coupled portion with (2,0) mode
        sin_fact    = np.sin(TWOPI*params['f_0']*(params['t_1']-params['t_0'])) #- np.sin(params['NRPMw_phi_fm'])
        dphi_mod    = params['D_2']*sin_fact/(TWOPI*params['f_0'])
        phi_pulse   = phi_bounce + TWOPI*params['f_2']*(params['t_1']-params['t_0']) + dphi_mod
        mu          = 1 - params['a_2']/np.sqrt(params['a_1']*params['a_3'])
        b_pulse     = np.log(params['a_3']/params['a_1'])/(params['t_3']-params['t_1'])

        h22 = h22 + sum([(1-mu/2.)*_fm_wavelet_func(freqs,
                                                    eta     = params['a_1']*np.exp(-1j*phi_pulse),
                                                    alpha   = 0.,
                                                    beta    = b_pulse-1j*TWOPI*params['f_2'],
                                                    tau     = params['t_3'] - params['t_1'],
                                                    tshift  = params['t_1'],
                                                    Omega   = TWOPI*params['f_0'],
                                                    Delta   = params['D_2'],
                                                    Gamma   = params['G_2'],
                                                    Phi     = PIHALF),
                         (mu/4.)*_fm_wavelet_func(freqs,
                                                  eta     = params['a_1']*np.exp(-1j*phi_pulse),
                                                  alpha   = 0.,
                                                  beta    = b_pulse-1j*TWOPI*(params['f_2']-params['f_0']),
                                                  tau     = params['t_3'] - params['t_1'],
                                                  tshift  = params['t_1'],
                                                  Omega   = TWOPI*params['f_0'],
                                                  Delta   = params['D_2'],
                                                  Gamma   = params['G_2'],
                                                  Phi     = PIHALF),
                         (mu/4.)*_fm_wavelet_func(freqs,
                                                  eta     = params['a_1']*np.exp(-1j*phi_pulse),
                                                  alpha   = 0.,
                                                  beta    = b_pulse-1j*TWOPI*(params['f_2']+params['f_0']),
                                                  tau     = params['t_3'] - params['t_1'],
                                                  tshift  = params['t_1'],
                                                  Omega   = TWOPI*params['f_0'],
                                                  Delta   = params['D_2'],
                                                  Gamma   = params['G_2'],
                                                  Phi     = PIHALF)])

    if params['NRPMw_t_coll'] > params['t_3'] :

        # h_rotating,
        # quasi-Lorentzian peak centered around f2
        phi_tail = phi_pulse + 2*np.pi*params['f_2']*(params['t_3']-params['t_1'])

        h22 = h22 + _fm_wavelet_func(freqs,
                                     eta     = params['a_3']*np.exp(-1j*phi_tail),
                                     alpha   = -1j*np.pi*params['NRPMw_df_2'],
                                     beta    = -params['B_2']-1j*TWOPI*params['f_2'],
                                     tau     = params['NRPMw_t_coll'] - params['t_3'],
                                     tshift  = params['t_3'],
                                     Omega   = TWOPI*params['f_0'],
                                     Delta   = params['D_2']*np.exp(-params['G_2']*(params['t_3'] - params['t_1'])),
                                     Gamma   = params['G_2'],
                                     Phi     = PIHALF)

    # compute hp,hc
    h22 *= _prefact*(params['mtot']**2./params['distance'])*np.exp(-1j*params['phi_ref'])
    return h22*(0.5*(1.+params['cosi']**2.)), h22*(params['cosi']*EmIPIHALF)

def nrpmw_wrapper(freqs, params):
    return NRPMw(freqs, params)

def nrpmw_recal_wrapper(freqs, params):
    return NRPMw(freqs, params, recalib=True)
