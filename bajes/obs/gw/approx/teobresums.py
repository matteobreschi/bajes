from __future__ import division, unicode_literals, absolute_import
import numpy as np

# Note: Running with TEOBResumS requires the EOBRun_module to be installed.
#
# The installation requires the C libray LibConfig and
# the following paths have to be exported:
# export TEOBRESUMS=/path/to/teobresums/C/
# export C_INCLUDE_PATH=/path/to/libconfig/build/include
# export LIBRARY_PATH=$LIBRARY_PATH:/path/to/libconfig/build/lib
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/libconfig/build/lib
#
# The TEOBResumS code can be downloaded from
# https://bitbucket.org/eob_ihes/teobresums/src
# and installed through:
# $ git checkout development
# $ make -f Makefile.TEOBResumS pywrap

import logging, warnings
logger = logging.getLogger(__name__)

__url__ = 'https://bitbucket.org/eob_ihes/teobresums/'
try:
    import EOBRun_module as EOB
except ImportError:
    warnings.warn("Unable to import TEOBResumS package. Please see related documentation at: {}".format(__url__))
    logger.warning("Unable to import TEOBResumS package. Please see related documentation at: {}".format(__url__))
    pass

from .... import MTSUN_SI
from ..utils import lambda_2_kappa
from .nrpm import NRPM
from .nrpmw import nrpmw_attach_wrapper, nrpmw_attach_recal_wrapper

def l_to_k(lmax, remove_ks = [], custom_modes=None):

    # Allow the user to select modes up to lmax only within a subset of specified modes
    if(custom_modes is not None):
        if(custom_modes=='Hyp'):
            #remove (l,m)=(3,2) from waveform, since not sane
            # trying to remove (5,5) for debugging
            modes_default = [[2,2], [2,1], [3,3], [4,4], [4,3]]
        else:
            raise ValueError('The requested option for custom modes does not exist.')

        modes = []
        for mode_x in modes_default:
            l_x, m_x = mode_x[0], mode_x[1]
            if(l_x < lmax): modes.append(mode_x)

    # Include all modes up to lmax
    else:
        all_l = np.arange(2, lmax+1)
        modes = np.concatenate([[[li,mi] for mi in range(1,li+1)] for li in all_l])

    k_modes = [int(x[0]*(x[0]-1)/2 + x[1]-2) for x in modes]

    if not(remove_ks==[]):
        for k_exlc in remove_ks:
            k_modes.remove(k_exlc)

    return k_modes

def additional_opts(params_teob, params):

    # set additional flags (present in params) in params_teob
    names = ['project_spins', 'postadiabatic_dynamics', 'use_flm', 'nqc', 'nqc_coefs_flx', 'nqc_coefs_hlm']

    for flag in names:
        if params.get(flag) is not None:
            params_teob[flag] = params[flag]

def teobresums(params_teob):
    """ Compute TEOBResumS time-domain waveform for spin-aligned compact binary coalescences.
        --------
        params_teob dictionary with the following
        key-arguments :

        M = total mass [solar masses]
        q = mass ratio [dimensionless]
        s1z = primary spin component along z axis [dimensionless]
        s2z = secondary spin component along z axis [dimensionless]
        Lam1 = primary dimensionless tidal parameter
        Lam2 = secondary dimensionless tidal parameter
        Deff = luminosity distance [Mpc]
        iota = inclination angle [rad]
        Fmin = initial frequency [Hz]
        phiRef = reference phase [rad]
        --------
        t = array
        hp = array, comples
        plus polarization
        hc = array, comples
        cross polarization
    """
    # Note: EOBRunTD returns (t, hp, hc)
    # Note: EOBRunFD returns (f, rhp, ihp, rhc, ihc)
    return EOB.EOBRunPy(params_teob)

def teobresums_wrapper(freqs, params):

    # unwrap lm modes
    if params['lmax'] == 0:
        modes = [1]
    else:
        modes = l_to_k(params['lmax'])

    # set TEOB dict
    params_teob = { 'M':                    params['mtot'],
                    'q':                    params['q'],
                    'chi1':                 params['s1z'],
                    'chi2':                 params['s2z'],
                    'chi1z':                params['s1z'],
                    'chi2z':                params['s2z'],
                    'LambdaAl2':            params['lambda1'],
                    'LambdaBl2':            params['lambda2'],
                    'distance':             params['distance'],
                    'inclination':          params['iota'],
                    'initial_frequency':    params['f_min'],
                    'coalescence_angle':    params['phi_ref'],
                    'use_geometric_units':  "no",
                    'output_hpc':           "no",
                    'interp_uniform_grid':  "yes",
                    'output_multipoles':    "no",
                    'use_mode_lm':          modes,
                    'srate':                params['srate'],
                    'srate_interp':         params['srate'],
                    'domain':               0
                    }

    if params['eccentricity'] != 0:
        params_teob['ecc'] = params['eccentricity']

    if params['s1x'] != 0:
        params_teob['chi1x'] = params['s1x']
    if params['s1y'] != 0:
        params_teob['chi1y'] = params['s1y']
    if params['s2x'] != 0:
        params_teob['chi2x'] = params['s2x']
    if params['s2y'] != 0:
        params_teob['chi2y'] = params['s2y']
    check = params['s1x']**2+params['s1y']**2+params['s2x']**2+params['s2y']**2
    if check > 1e-7:
        params_teob['use_spins'] = 2

    # check for additional options
    additional_opts(params_teob, params)

    t , hp , hc     = teobresums(params_teob)
    return hp , hc

def teobresums_a6cfree_wrapper(freqs, params):

    # unwrap lm modes
    if params['lmax'] == 0:
        modes = [1]
    else:
        modes = l_to_k(params['lmax'])

    # set TEOB dict
    params_teob = { 'M':                    params['mtot'],
                    'q':                    params['q'],
                    'chi1':                 params['s1z'],
                    'chi2':                 params['s2z'],
                    'chi1z':                params['s1z'],
                    'chi2z':                params['s2z'],
                    'LambdaAl2':            params['lambda1'],
                    'LambdaBl2':            params['lambda2'],
                    'distance':             params['distance'],
                    'inclination':          params['iota'],
                    'initial_frequency':    params['f_min'],
                    'coalescence_angle':    params['phi_ref'],
                    'use_geometric_units':  "no",
                    'output_hpc':           "no",
                    'interp_uniform_grid':  "yes",
                    'output_multipoles':    "no",
                    'use_mode_lm':          modes,
                    'srate':                params['srate'],
                    'srate_interp':         params['srate'],
                    'domain':               0,
                    'a6c':                  params['TEOBResumS_a6c']
                    }

    if params['eccentricity'] != 0:
        params_teob['ecc'] = params['eccentricity']

    if params['s1x'] != 0:
        params_teob['chi1x'] = params['s1x']
    if params['s1y'] != 0:
        params_teob['chi1y'] = params['s1y']
    if params['s2x'] != 0:
        params_teob['chi2x'] = params['s2x']
    if params['s2y'] != 0:
        params_teob['chi2y'] = params['s2y']
    check = params['s1x']**2+params['s1y']**2+params['s2x']**2+params['s2y']**2
    if check > 1e-7:
        params_teob['use_spins'] = 2

    # check for additional options
    additional_opts(params_teob, params)

    t , hp , hc     = teobresums(params_teob)
    return hp , hc

# Requires the teob eccentric branch
def teobresums_ecc_wrapper(freqs, params):
    # unwrap lm modes
    if params['lmax'] == 0:
        modes = [1]
    else:
        modes = l_to_k(params['lmax'])

    # set TEOB dict
    params_teob = { 'M':                    params['mtot'],
                    'q':                    params['q'],
                    'chi1':                 params['s1z'],
                    'chi2':                 params['s2z'],
                    'chi1z':                params['s1z'],
                    'chi2z':                params['s2z'],
                    'Lambda1':              params['lambda1'],
                    'Lambda2':              params['lambda2'],
                    'distance':             params['distance'],
                    'inclination':          params['iota'],
                    'initial_frequency':    params['f_min'],
                    'coalescence_angle':    params['phi_ref'],
                    'use_geometric_units':  0,
                    'output_hpc':           0,
                    'interp_uniform_grid':  1,
                    'output_multipoles':    0,
                    'use_mode_lm':          modes,
                    'srate':                params['srate'],
                    'srate_interp':         params['srate'],
                    'domain':               0
                    }

    if params['eccentricity'] != 0:
        params_teob['ecc'] = params['eccentricity']

    if params['s1x'] != 0:
        params_teob['chi1x'] = params['s1x']
    if params['s1y'] != 0:
        params_teob['chi1y'] = params['s1y']
    if params['s2x'] != 0:
        params_teob['chi2x'] = params['s2x']
    if params['s2y'] != 0:
        params_teob['chi2y'] = params['s2y']
    check = params['s1x']**2+params['s1y']**2+params['s2x']**2+params['s2y']**2
    if check > 1e-7:
        params_teob['use_spins'] = 2

    # check for additional options
    additional_opts(params_teob, params)

    t , hp , hc     = teobresums(params_teob)
    return hp , hc

def teobresums_hyperb_wrapper(freqs, params):

    # unwrap lm modes
    if params['lmax'] == 0: k_modes = [1] # 22-only
    else:                   k_modes = l_to_k(params['lmax'], custom_modes='Hyp') # Keep only modes we trust

    # Auxiliary parameters
    a0 = (params['s1z'] * params['q'] + params['s2z'])/(1+params['q'])
    nu = params['q']/np.power(1+params['q'], 2.)

    # Initial conditions
    r = 1500. # Asymptotic radius at which the evolution is started.

    # Restrict to regions far from direct capture
    # For the non-spinning case this is done through the prior,
    # while the spinning case has too much variability to impose the constraint through fixed pphi0 bounds.
    if(params['s1z']==0.0 and params['s2z']==0.0): pphi_lso_low_limit = 1.0
    else:                                          pphi_lso_low_limit = 1.15
    pphi_lso = EOB.pph_lso_spin_py(nu, a0)
    if (params['angmom'] < pphi_lso_low_limit*pphi_lso): return [None], [None]

    # compute E_min and E_max
    Emn, Emx = EnergyLimits(r, params['q'], params['angmom'], params['s1z'], params['s2z'])

    if (params['Eprior']):
        if(   (params['Eprior']=='Constrained')   and ( (params['energy'] < Emn) or (params['energy'] > Emx) ) ):  return [None], [None]
        elif( (params['Eprior']=='Unconstrained') and ( (params['energy'] < Emn)                             ) ):  return [None], [None]

    # set TEOB dict
    params_teob = {
                    # Standard source parameters
                    'M':                   params['mtot']    ,
                    'q':                   params['q']       ,
                    'chi1':                params['s1z']     ,
                    'chi2':                params['s2z']     ,
                    'Lambda1':             params['lambda1'] ,
                    'Lambda2':             params['lambda2'] ,
                    'distance':            params['distance'],
                    'inclination':         params['iota']    ,
                    'coalescence_angle':   params['phi_ref'] ,

                    # Hyperbolic parameters
                    'ecc':                 0.0               ,
                    'r0':                  r                 ,
                    'r_hyp':               r                 ,
                    'j_hyp':               params['angmom']  ,
                    'H_hyp':               params['energy']  ,

                    # Waveform generation parameters
                    'use_geometric_units': 0                 ,
                    'output_hpc':          0                 ,
                    'output_multipoles':   0                 ,
                    'use_mode_lm':         k_modes           ,
                    'domain':              0                 ,
                    'arg_out':             0                 ,
                    'initial_frequency':   params['f_min']   ,
                    'srate':               params['srate']   ,
                    'srate_interp':        params['srate']   ,
                    'dt':                  0.5               ,
                    'dt_interp':           0.5               ,
                    'ode_tmax':            20e4              ,
                    'interp_uniform_grid': 2                 ,
            }

    # Turn NQCs off
    if not(params['nqc-TEOBHyp']):
        params_teob['nqc']           = 2  # set NQCs manually
        params_teob['nqc_coefs_hlm'] = 0  # turn NQC off for hlm
        params_teob['nqc_coefs_flx'] = 0  # turn NQC off for flx

    t, hp, hc = teobresums(params_teob)

    if(np.any(np.isnan(hp)) or np.any(np.isnan(hc))):
        logger.warning('Nans in the waveform, with the configuration: {}. Returning None and skipping sample.'.format(params_teob))
        return [None], [None]
    if(np.any(np.isinf(hp)) or np.any(np.isinf(hc))):
        logger.warning('Infinities in the waveform, with the configuration: {}. Returning None and skipping sample.'.format(params_teob))
        return [None], [None]

    return hp, hc

def teobresums_spa_wrapper(freqs, params):

    #unwrap lm modes
    if params['lmax'] == 0:
        modes = [1]
    else:
        modes = l_to_k(params['lmax'])

    # set TEOB dict
    params_teob = { 'M':                    params['mtot'],
                    'q':                    params['q'],
                    'chi1':                 params['s1z'],
                    'chi2':                 params['s2z'],
                    'chi1z':                params['s1z'],
                    'chi2z':                params['s2z'],
                    'LambdaAl2':            params['lambda1'],
                    'LambdaBl2':            params['lambda2'],
                    'distance':             params['distance'],
                    'inclination':          params['iota'],
                    'coalescence_angle':    params['phi_ref'],
                    'srate':                params['srate'],
                    'srate_interp':         params['srate'],
                    'use_geometric_units':  "no",
                    'output_hpc':           "no",
                    'output_multipoles':    "no",
                    'use_mode_lm':          modes,
                    'domain':               1,
                    'interp_freqs':         "yes",
                    'freqs':                freqs.tolist(),
                    'initial_frequency':    params['f_min']
                    }

    if params['eccentricity'] != 0:
        params_teob['ecc'] = params['eccentricity']

    if params['s1x'] != 0:
        params_teob['chi1x'] = params['s1x']
    if params['s1y'] != 0:
        params_teob['chi1y'] = params['s1y']
    if params['s2x'] != 0:
        params_teob['chi2x'] = params['s2x']
    if params['s2y'] != 0:
        params_teob['chi2y'] = params['s2y']
    check = params['s1x']**2+params['s1y']**2+params['s2x']**2+params['s2y']**2
    if check > 1e-7:
        params_teob['use_spins'] = 2

    # check for additional options
    additional_opts(params_teob, params)

    f , rhplus, ihplus, rhcross, ihcross = teobresums(params_teob)
    return rhplus-1j*ihplus, rhcross-1j*ihcross

def teobresums_spa_nrpmw_wrapper(freqs, params):

    #unwrap lm modes
    modes = [1]
    if params['lmax'] != 0:
        warnings.warn("TEOBResumSPA_NRPMw model provides only (2,2) mode")

    # set TEOB dict
    params_teob = { 'M':                    params['mtot'],
                    'q':                    params['q'],
                    'chi1':                 params['s1z'],
                    'chi2':                 params['s2z'],
                    'chi1z':                params['s1z'],
                    'chi2z':                params['s2z'],
                    'LambdaAl2':            params['lambda1'],
                    'LambdaBl2':            params['lambda2'],
                    'distance':             params['distance'],
                    'inclination':          params['iota'],
                    'coalescence_angle':    params['phi_ref'],
                    'srate':                params['srate'],
                    'srate_interp':         params['srate'],
                    'use_geometric_units':  "no",
                    'output_hpc':           "no",
                    'output_multipoles':    "no",
                    'use_mode_lm':          modes,
                    'domain':               1,
                    'interp_freqs':         "yes",
                    'freqs':                freqs.tolist(),
                    'initial_frequency':    params['f_min']
                    }

    if params['eccentricity'] != 0:
        params_teob['ecc'] = params['eccentricity']

    if params['s1x'] != 0:
        params_teob['chi1x'] = params['s1x']
    if params['s1y'] != 0:
        params_teob['chi1y'] = params['s1y']
    if params['s2x'] != 0:
        params_teob['chi2x'] = params['s2x']
    if params['s2y'] != 0:
        params_teob['chi2y'] = params['s2y']
    check = params['s1x']**2+params['s1y']**2+params['s2x']**2+params['s2y']**2
    if check > 1e-7:
        params_teob['use_spins'] = 2

    # check for additional options
    additional_opts(params_teob, params)

    # compute EOB waveform
    f, re_hp, im_hp, re_hc, im_hc = teobresums(params_teob)
    hp_eob, hc_eob   = re_hp-1j*im_hp, re_hc-1j*im_hc
    # compute PM waveform
    hp_pm, hc_pm = nrpmw_attach_wrapper(freqs, params)
    return hp_eob+hp_pm , hc_eob+hc_pm

def teobresums_nrpm_wrapper(freqs, params):

    # generate TEOB
    hp_eob, hc_eob = teobresums_wrapper(freqs, params)
    h_eob = hp_eob - 1j*hc_eob

    # estimate merger frequency
    phi_last    = np.angle(h_eob[-1])
    f_merg      = np.abs(np.gradient(np.unwrap(np.angle(h_eob[-100:])))*params['srate'])[-1]/(2*np.pi)

    # generate NRPM
    kappa2T = lambda_2_kappa(params['mtot']/(1.+1./params['q']),
                             params['mtot']/(1.+ params['q']),
                             params['lambda1'], params['lambda2'])

    if kappa2T < 60 :
        return hp_eob, hc_eob

    hp_pm, hc_pm = NRPM(params['srate'], params['seglen'], params['mtot'], params['q'], kappa2T,
                        params['distance'], params['iota'], phi_last,
                        f_merg = f_merg, alpha = None, phi_kick = None)

    h_pm = hp_pm - 1j*hc_pm

    # remove tail before merger and rescale amplitude
    amp_fact = np.max(np.abs(h_eob))/np.max(np.abs(h_pm))
    h_pm = h_pm[np.argmax(np.abs(h_pm))+1:] * amp_fact

    h = np.append(h_eob , h_pm)
    return np.real(h), -np.imag(h)

def teobresums_spa_nrpmw_recal_wrapper(freqs, params):

    #unwrap lm modes
    modes = [1]
    if params['lmax'] != 0:
        warnings.warn("TEOBResumSPA_NRPMw model provides only (2,2) mode")

    # set TEOB dict
    params_teob = { 'M':                    params['mtot'],
                    'q':                    params['q'],
                    'chi1':                 params['s1z'],
                    'chi2':                 params['s2z'],
                    'chi1z':                params['s1z'],
                    'chi2z':                params['s2z'],
                    'LambdaAl2':            params['lambda1'],
                    'LambdaBl2':            params['lambda2'],
                    'distance':             params['distance'],
                    'inclination':          params['iota'],
                    'coalescence_angle':    params['phi_ref'],
                    'srate':                params['srate'],
                    'srate_interp':         params['srate'],
                    'use_geometric_units':  "no",
                    'output_hpc':           "no",
                    'output_multipoles':    "no",
                    'use_mode_lm':          modes,
                    'domain':               1,
                    'interp_freqs':         "yes",
                    'freqs':                freqs.tolist(),
                    'initial_frequency':    params['f_min']
                    }

    if params['eccentricity'] != 0:
        params_teob['ecc'] = params['eccentricity']

    if params['s1x'] != 0:
        params_teob['chi1x'] = params['s1x']
    if params['s1y'] != 0:
        params_teob['chi1y'] = params['s1y']
    if params['s2x'] != 0:
        params_teob['chi2x'] = params['s2x']
    if params['s2y'] != 0:
        params_teob['chi2y'] = params['s2y']
    check = params['s1x']**2+params['s1y']**2+params['s2x']**2+params['s2y']**2
    if check > 1e-7:
        params_teob['use_spins'] = 2

    # check for additional options
    additional_opts(params_teob, params)

    # compute EOB waveform
    f, re_hp, im_hp, re_hc, im_hc = teobresums(params_teob)
    hp_eob, hc_eob   = re_hp-1j*im_hp, re_hc-1j*im_hc
    # compute PM waveform
    hp_pm, hc_pm = nrpmw_attach_recal_wrapper(freqs, params)
    return hp_eob+hp_pm , hc_eob+hc_pm

def Espin(r, pph, q, chi1, chi2):
    # New energy potential energy function with the full spin dependence.
    hatH = EOB.eob_ham_s_py(r, q, pph, 0., chi1, chi2)
    nu   = q/(1+q)**2
    E0   = nu*hatH[0]
    return E0

def EnergyLimits(rmx, q, pph_hyp, chi1, chi2, N=100000):

    # r_min is the smallest radius at which the potenatial can peak.
    # For |chi|<0.5 and pphi < 1.55 * pphi_LSO, r_min > 1.5 (equal mass case, where r_min is smaller),
    # so we set 1.3 to be conservative and ignore nans.
    # Without spin, same considerations with 1.5
    if chi1!=0 or chi2!=0: rmin = 1.3
    else:                  rmin = 1.5

    x    = np.linspace(rmin, rmx, N)
    E0   = list(map(lambda i : Espin(i, pph_hyp, q, chi1, chi2), x))
    Emin = Espin(rmx, pph_hyp, q, chi1, chi2)
    # Determine the max energy allowed. For large q, A will go below zero, so ignore those values by removing nans.
    Emx  = np.nanmax(E0)

    return Emin, Emx
