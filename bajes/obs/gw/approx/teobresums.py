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

try:
    import EOBRun_module as EOB
except Exception:
    pass

def l_to_k(lmax):
    all_l = np.arange(2, lmax+1)
    modes = np.concatenate([[[li,mi] for mi in range(1,li+1)] for li in all_l])
    return [int(x[0]*(x[0]-1)/2 + x[1]-2) for x in modes]

def  additional_opts(params_teob, params):

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
                    'Lambda1':              params['lambda1'],
                    'Lambda2':              params['lambda2'],
                    'distance':             params['distance'],
                    'inclination':          params['iota'],
                    'initial_frequency':    params['f_min'],
                    'coalescence_angle':    params['phi_ref'],
                    'use_geometric_units':  0,
                    'output_hpc':           0,
                    'interp_uniform_grid':  2,
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
    if params['lmax'] == 0:
        modes = [1]
    else:
        modes = l_to_k(params['lmax'])

#    # compute J_lso
    nu = params['q']/np.power(1+params['q'], 2.)
    pphi_lso = EOB.pph_lso_orbital_py(nu)
    r = 1500. 

    if params['angmom'] >= pphi_lso:
#
        # compute E_min and E_max
        Emn, Emx, Einfl = EnergyLimits(r, params['q'], params['angmom'], params['s1z'], params['s2z'])
        if params['energy'] >= Emn and params['energy'] <= Emx:

    # set TEOB dict
          params_teob = { 'M':              params['mtot'],
                    'q':                    params['q'],
                    'chi1':                 params['s1z'],
                    'chi2':                 params['s2z'],
                    'Lambda1':              params['lambda1'],
                    'Lambda2':              params['lambda2'],
                    'distance':             params['distance'],
                    'inclination':          params['iota'],
                    'initial_frequency':    params['f_min'],
                    'coalescence_angle':    params['phi_ref'],
                    'use_geometric_units':  0,
                    'output_hpc':           0,
                    'interp_uniform_grid':  2,
                    'output_multipoles':    0,
                    'use_mode_lm':          modes,
                    'srate':                params['srate'],
                    'srate_interp':         params['srate'],
                    'domain':               0,
                    'dt':                   0.5,
                    'dt_interp':            0.5,
                    'arg_out':              0,
                    'r0':                   r,
                    'ecc':                  0.18,
                    'j_hyp':                params['angmom'],
                    'r_hyp':                r,
                    'H_hyp':                params['energy'],
                    'ode_tmax':             20e4,
                    'nqc':                  2,  # set NQCs manually
                    'nqc_coefs_hlm':        0,  # turn NQC off for hlm
                    'nqc_coefs_flx':        0   # turn NQC off for flx

                    }

          t , hp , hc     = teobresums(params_teob)
          return hp , hc
        else:
          return [None], [None]
    else:
      return [None], [None]

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
                    'Lambda1':              params['lambda1'],
                    'Lambda2':              params['lambda2'],
                    'distance':             params['distance'],
                    'inclination':          params['iota'],
                    'coalescence_angle':    params['phi_ref'],
                    'srate':                np.max(freqs)*2,
                    'interp_FD_waveform':   1,
                    'use_geometric_units':  0,
                    'output_hpc':           0,
                    'output_multipoles':    0,
                    'use_mode_lm':          modes,
                    'domain':               1,
                    'interp_freqs':         1,
                    'freqs':                freqs.tolist(),
                    }

    f , rhplus, ihplus, rhcross, ihcross = teobresums(params_teob)
    return rhplus-1j*ihplus, rhcross-1j*ihcross

def teobresums_nrpm_wrapper(freqs, params):

    # generate TEOB
    hp_eob, hc_eob = teobresums_wrapper(freqs, params)
    h_eob = hp_eob - 1j*hc_eob
    
    # estimate merger frequency
    phi_last    = np.angle(h_eob[-1])
    f_merg      = np.abs(np.gradient(np.unwrap(np.angle(h_eob[-100:])))*params['srate'])[-1]/(2*np.pi)

    # generate NRPM
    from .nrpm import NRPM
    from ..utils import lambda_2_kappa
    
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

def D4(f, dx):
    
    n       = len(f)
    oo12dx  = 1./(12*dx)
    df      = list(map(lambda i: (8.*(f[i+1]-f[i-1]) - f[i+2] + f[i-2])*oo12dx, range(2, n-2)))

    i = 0
    df0 = (-25.*f[i] + 48.*f[i+1] - 36.*f[i+2] + 16.*f[i+3] - 3.*f[i+4])*oo12dx
    i = 1
    df1 = (-3.*f[i-1] - 10.*f[i] + 18.*f[i+1] - 6.*f[i+2] + f[i+3])*oo12dx
    i = n-2
    df_2 = - (-3.*f[i+1] - 10.*f[i] + 18.*f[i-1] - 6.*f[i-2] + f[i-3])*oo12dx
    i = n-1
    df_1 = - (-25.*f[i] + 48.*f[i-1] - 36.*f[i-2] + 16.*f[i-3] - 3.*f[i-4])*oo12dx

    return np.concatenate([[df0], [df1], df, [df_2], [df_1]])

def E(r, pph, nu):
    A, dA, d2A  = EOB.eob_metric_A5PNlog_py(r, nu)
    Heff0       = np.sqrt(A*(1+(pph/r)**2))
    E0          = np.sqrt(1 + 2*nu*(Heff0-1))
    return E0

def Espin(r, pph, q, chi1, chi2):
    # New energy potential energy function with the full spin dependence.
    hatH = EOB.eob_ham_s_py(r, q, pph, 0., chi1, chi2)
    nu   = q/(1+q)**2
    E0   = nu*hatH[0]
    return E0

def EnergyLimits(rmx, q, pph_hyp, chi1, chi2):

    if chi1!=0 or chi2!=0:
        # important: rmin should be smaller with spin
        rmin = 1.1
    else:
        rmin = 1.3

    x    = np.linspace(rmin,rmx+10, 100000)
    E0   = list(map(lambda i : Espin(i, pph_hyp, q, chi1, chi2), x))
    Emin = Espin(rmx, pph_hyp, q, chi1, chi2)

    dx   = x[1]-x[0]
    dE0  = D4(E0,  dx)
    d2E0 = D4(dE0, dx)

    #-----------------------------------
    #  determine the max energy allowed
    #-----------------------------------

    Emx = np.max(E0)

    #---------------------------------------
    # determine the inflection point of the
    # potential energy E0.
    #--------------------------------------

    jflex = np.where(d2E0 >= 0)[0][0] #FIXME: this needs fixing
    Einfl = E0[jflex]

    return Emin, Emx, Einfl
