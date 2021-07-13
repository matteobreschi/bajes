from __future__ import division, unicode_literals, absolute_import
import numpy as np
from scipy.special import wofz, factorial, binom

from . import _nrpmw_fits
from ..... import MTSUN_SI, MRSUN_SI, PARSEC_SI

import logging
logger = logging.getLogger(__name__)

PIHALF      = np.pi/2.
TWOPI       = 2.*np.pi
SQRTPI      = np.sqrt(np.pi)
EmIPIHALF   = np.exp(-1j*PIHALF)

SPHARM_22   = np.sqrt(5./(4.*np.pi))
_prefact    = SPHARM_22*MRSUN_SI*MTSUN_SI/1e6/PARSEC_SI

#####################################################
#                                                   #
#           Baseline wavelet functions              #
#                                                   #
#####################################################

def __nan_in_array(x):
    if any(np.isnan(x)):
        return True
    else:
        return False

def _wavelet_integral_extremes(x,b):
    """
        Compute Gaussian integral given the extremal values
    """
    # compute wofz
    b_plus_x = b+x
    d1 = wofz(b_plus_x)
    d2 = wofz(x)
    out = -0.5j*SQRTPI*(np.exp(b*b + 2.*x*b)*d1 - d2)
    if __nan_in_array(out):
        logger.warning("NRPMw is unable to compute wavelet integral, returning zero for this wavelet component")
        return 0.
    else:
        return out

def _wavelet_func(freq, eta, alpha, beta, tau, tshift=0):
    """
        Frequency-domain representation of

            W(t) = eta * exp( alpha * t**2 + beta*t)

        with eta, alpha, beta in C integrated over t in [0, tau]
        and shifted of a time tshift

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
    """

    if tau <= 0:
        return 0.

    imalpha = np.imag(alpha)

    if np.abs(alpha) < 1e-16:

        # exponential case, alpha = 0

        eta     = complex(eta)
        beta    = complex(beta)

        x1      = beta-1j*TWOPI*freq
        c1      = 0.5*eta*(np.exp(x1*tau) - 1. )/x1

        beta    = np.conj(beta)
        eta     = np.conj(eta)

        x1      = beta-1j*TWOPI*freq
        c2      = 0.5*eta*(np.exp(x1*tau) - 1. )/x1

        return (c1 + c2) * np.exp(-1j*TWOPI*freq*tshift)

    elif np.abs(imalpha) < 1e-4 and imalpha > 0:

        # gaussian case, im(alpha) small and positive

        eta     = complex(eta)
        alpha   = complex(alpha)
        beta    = complex(beta)
        sqrta   = np.sqrt(alpha)

        x1      = 0.5*(beta-2.j*np.pi*freq)/sqrta
        b1      = sqrta*tau
        c1      = 0.5*eta/sqrta

        return c1 * _wavelet_integral_extremes(x1,b1) * np.exp(-1j*TWOPI*freq*tshift)

    elif np.abs(imalpha) < 1e-4 and imalpha < 0:

        # gaussian case, im(alpha) small and negative

        alpha   = np.conj(complex(alpha))
        beta    = np.conj(complex(beta))
        eta     = np.conj(complex(eta))
        sqrta   = np.sqrt(alpha)

        x2      = 0.5*(beta-2.j*np.pi*freq)/sqrta
        b2      = sqrta*tau
        c2      = 0.5*eta/sqrta

        return c2 * _wavelet_integral_extremes(x2,b2) * np.exp(-1j*TWOPI*freq*tshift)

    else:

        # gaussian case, general

        eta     = complex(eta)
        alpha   = complex(alpha)
        beta    = complex(beta)
        sqrta   = np.sqrt(alpha)

        x1      = 0.5*(beta-2.j*np.pi*freq)/sqrta
        b1      = sqrta*tau
        c1      = 0.5*eta/sqrta

        alpha   = np.conj(alpha)
        beta    = np.conj(beta)
        eta     = np.conj(eta)
        sqrta   = np.sqrt(alpha)

        x2      = 0.5*(beta-2.j*np.pi*freq)/sqrta
        b2      = sqrta*tau
        c2      = 0.5*eta/sqrta

        return (c1 * _wavelet_integral_extremes(x1,b1) + c2 * _wavelet_integral_extremes(x2,b2)) * np.exp(-1j*TWOPI*freq*tshift)

def _fm_wavelet_func(freq, eta, alpha, beta, tau, tshift, Omega, Delta, Gamma, Phi):
    """
        Frequency-domain representation of frequency-modulated wavelet;

            W(t) = eta * exp( alpha * t**2 + beta*t ) * exp( -i F(t) )

        with eta, alpha, beta in C integrated over t in [0, tau]
        and shifted of a time tshift.
        F(t) is a sinusoidal+expontial function that defines the frequency modulations (FMs)
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

    if np.abs(Delta) > 1e-16 and tau > 0:

        # set constants
        nu          = -Gamma - 1j*Omega
        d           = -0.5*Delta/np.abs(nu)**2
        phi_extra   = np.exp(1j*2.*d*(Gamma*np.sin(Phi)+Omega*np.cos(Phi)))
        nmax        = min(max(1,int(2.*(1.+np.abs(Delta/Omega)))),12)

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
        h22 *= _prefact*(params['mtot']**2./params['distance'])*np.exp(-1j*(params['phi_ref'] + freqs*(TWOPI*params['time_shift']/(MTSUN_SI*params['mtot']))))
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
                                     Delta   = params['NRPMw_delta_fm'],
                                     Gamma   = 0.,
                                     Phi     = -PIHALF+params['NRPMw_phi_fm'])

    if params['NRPMw_t_coll'] > params['t_1'] :

        # h_pulse,
        # coupled portion with (2,0) mode
        sin_fact    = np.sin(TWOPI*params['f_0']*(params['t_1']-params['t_0'])+params['NRPMw_phi_fm']) - np.sin(params['NRPMw_phi_fm'])
        dphi_mod    = params['NRPMw_delta_fm']*sin_fact/(TWOPI*params['f_0'])
        phi_pulse   = phi_bounce + TWOPI*params['f_2']*(params['t_1']-params['t_0']) + dphi_mod
        mu          = 1 - params['a_2']/np.sqrt(params['a_1']*params['a_3'])
        b_pulse     = np.log(params['a_3']/params['a_1'])/(params['t_3']-params['t_1'])

        h22 = h22 + sum([(1-mu/2.)*_fm_wavelet_func(freqs,
                                                    eta     = params['a_1']*np.exp(-1j*phi_pulse),
                                                    alpha   = 0,
                                                    beta    = b_pulse-1j*TWOPI*params['f_2'],
                                                    tau     = params['t_3'] - params['t_1'],
                                                    tshift  = params['t_1'],
                                                    Omega   = TWOPI*params['f_0'],
                                                    Delta   = params['NRPMw_delta_fm'],
                                                    Gamma   = params['NRPMw_gamma_fm'],
                                                    Phi     = PIHALF+params['NRPMw_phi_fm']),
                         (mu/4.)*_fm_wavelet_func(freqs,
                                                  eta     = params['a_1']*np.exp(-1j*phi_pulse),
                                                  alpha   = 0,
                                                  beta    = b_pulse-1j*TWOPI*(params['f_2']-params['f_0']),
                                                  tau     = params['t_3'] - params['t_1'],
                                                  tshift  = params['t_1'],
                                                  Omega   = TWOPI*params['f_0'],
                                                  Delta   = params['NRPMw_delta_fm'],
                                                  Gamma   = params['NRPMw_gamma_fm'],
                                                  Phi     = PIHALF+params['NRPMw_phi_fm']),
                         (mu/4.)*_fm_wavelet_func(freqs,
                                                  eta     = params['a_1']*np.exp(-1j*phi_pulse),
                                                  alpha   = 0,
                                                  beta    = b_pulse-1j*TWOPI*(params['f_2']+params['f_0']),
                                                  tau     = params['t_3'] - params['t_1'],
                                                  tshift  = params['t_1'],
                                                  Omega   = TWOPI*params['f_0'],
                                                  Delta   = params['NRPMw_delta_fm'],
                                                  Gamma   = params['NRPMw_gamma_fm'],
                                                  Phi     = PIHALF+params['NRPMw_phi_fm'])])

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
                                     Delta   = params['NRPMw_delta_fm']*np.exp(-params['NRPMw_gamma_fm']*(params['t_3'] - params['t_1'])),
                                     Gamma   = params['NRPMw_gamma_fm'],
                                     Phi     = PIHALF+params['NRPMw_phi_fm'])

    # compute hp,hc
    h22 *= _prefact*(params['mtot']**2./params['distance'])*np.exp(-1j*(params['phi_ref'] + freqs*(TWOPI*params['time_shift']/(MTSUN_SI*params['mtot']))))
    return h22*(0.5*(1.+params['cosi']**2.)), h22*(params['cosi']*EmIPIHALF)
