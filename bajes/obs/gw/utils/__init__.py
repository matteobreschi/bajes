#!/usr/bin/env python
from __future__ import division, unicode_literals, absolute_import
__import__("pkg_resources").declare_namespace(__name__)

import os, sys
import numpy as np

from scipy.signal import decimate

from ..noise import get_design_sensitivity, get_event_sensitivity, get_event_calibration
from ..strain import fft, ifft

def lambda_2_kappa(M1,M2,L1,L2):
    Mt = M1 + M2
    k1 = 3. * L1 * (M2/M1) * (M1/Mt)**5
    k2 = 3. * L2 * (M1/M2) * (M2/Mt)**5
    return k1 + k2

def compute_quadrupole_yy(lam):
    # Compute quadrupole coefficient from Lambda using the chi precessing spin parameter (for given 3-dim spin vectors)
    if lam <= 0. : return 1.
    loglam = np.log(lam)
    logCQ = 0.194 + 0.0936*loglam + 0.0474*loglam**2 - 4.21e-3*loglam**3 + 1.23e-4*loglam**4
    return np.exp(logCQ)

def q_2_eta(q):
    return q/((1+q)**2)

def mcm_to_m2(m1,mc):
    c   = [m1**3 , 0 , -mc**5 , -mc**5*m1]
    m2  = np.real(np.roots(c))
    return float(m2[np.where(m2>=0)])

def m2q_to_m1(m2,q):
    return q*m2

def m1q_to_m2(m1,q):
    return m1/q

def mcq_to_m1(mc,q):
    return mc*(q*q*(1.+q))**0.2

def mcq_to_m2(mc,q):
    return mc*((1.+q)/(q**3.))**0.2

def eta_2_q(eta):
    """ Compute mass ratio q = m1/m2 from symmetric mass ratio
        --------
        eta = symmetric mass ratio (0 < eta <= 1/4)

        if nu > 1/4 then q < 1, which is unphysical.
        Equal-mass case (q = 1) is returned.
    """
    if not (np.any(eta>0.25) or np.any(eta <=0)):
        return (1.-2.*eta + np.sqrt(1.-4.*eta))/2./eta
    else:
        raise ValueError("Symmetric mass ratio must be greater than 0 and lower then 1/4.")

def compute_chi_eff(m1,m2,s1,s2):
    """ Compute chi effective spin parameter (for a given component)
        --------
        m1 = primary mass component [solar masses]
        m2 = secondary mass component [solar masses]
        s1 = primary spin z-component [dimensionless]
        s2 = secondary spin z-component [dimensionless]
    """
    return (m1*s1+m2*s2)/(m1+m2)

def compute_chi_prec(m1,m2,s1,s2,tilt1,tilt2):
    """ Compute chi precessing spin parameter (for given 3-dim spin vectors)
        --------
        m1 = primary mass component [solar masses]
        m2 = secondary mass component [solar masses]
        s1 = primary spin megnitude [dimensionless]
        s2 = secondary spin megnitude [dimensionless]
        tilt1 = primary spin tilt [rad]
        tilt2 = secondary spin tilt [rad]
    """

    s1_perp = np.abs(s1*np.sin(tilt1))
    s2_perp = np.abs(s2*np.sin(tilt2))
    one_q   = m2/m1

    # check that m1>=m2, otherwise switch
    if one_q > 1. :
        one_q = 1./one_q
        s1_perp, s2_perp = s2_perp, s1_perp

    return np.max([s1_perp , s2_perp*one_q*(4.*one_q+3.)/(3.*one_q+4.)])

def compute_tidal_components(m1, m2 , Lamt , dLam):
    """ Compute tides components from masses and (Lambda Tilde, delta Lambda)
        --------
        m1 = primary mass component [solar masses]
        m2 = secondary mass component [solar masses]
        Lamt    = tidal combination @ 5PN [dimensionless]
        dLam   = tidal combination @ 6PN [dimensionless]
        """
    M       = m1 + m2
    q       = m1/m2
    eta     = q/((1.+q)*(1.+q))
    X       = np.sqrt(1.-4.*eta)
    m1_4    = m1**4.
    m2_4    = m2**4.
    M4      = M**4.
    M5      = M4 * M
    a       = (1690.*eta/1319. - 4843./1319.)
    b       = (6162.*X/1319.)
    ap      = a + b
    am      = a - b
    A       = (m1 + (12.*m2))*m1_4/M5
    B       = (m2 + (12.*m1))*m2_4/M5

    num2    = ((13./(16.*A))*Lamt - M4*dLam/m1_4/ap)
    den2    = am*m2_4/ap/m1_4 + B/A

    lambda2 = num2/den2
    lambda1 = (13.*Lamt/16. - B*lambda2)/A

    if lambda1 < 0 : lambda1 = 0.
    if lambda2 < 0 : lambda2 = 0.
    return lambda1 , lambda2

def compute_lambda_tilde(m1, m2 ,l1 , l2):
    """ Compute Lambda Tilde from masses and tides components
        --------
        m1 = primary mass component [solar masses]
        m2 = secondary mass component [solar masses]
        l1 = primary tidal component [dimensionless]
        l2 = secondary tidal component [dimensionless]
        """
    M       = m1 + m2
    m1_4    = m1**4.
    m2_4    = m2**4.
    M5      = M**5.
    comb1   = m1 + 12.*m2
    comb2   = m2 + 12.*m1
    return (16./13.)*(comb1*m1_4*l1 + comb2*m2_4*l2)/M5

def compute_delta_lambda(m1, m2 ,l1 , l2):
    """ Compute delta Lambda Tilde from masses and tides components
        --------
        m1 = primary mass component [solar masses]
        m2 = secondary mass component [solar masses]
        l1 = primary tidal component [dimensionless]
        l2 = secondary tidal component [dimensionless]
        """
    M       = m1+m2
    q       = m1/m2
    eta     = q/((1.+q)*(1.+q))
    X       = np.sqrt(1.-4.*eta)
    m1_4    = m1**4.
    m2_4    = m2**4.
    M4      = M**4.
    comb1   = (1690.*eta/1319. - 4843./1319.)*(m1_4*l1 - m2_4*l2)/M4
    comb2   = (6162.*X/1319.)*(m1_4*l1 + m2_4*l2)/M4
    return comb1 + comb2

def tdwf_2_fdwf(freqs , h, dt):

    # compute fft
    fr , h_fft = fft(h, dt)

    # interpolate amplitude and phase
    amp_interp  = np.interp(freqs, fr, np.abs(h_fft))
    phi_interp  = np.interp(freqs, fr, np.unwrap(np.angle(h_fft)))

    # return FD-WF
    return amp_interp * np.exp(1j * phi_interp)

def fdwf_2_tdwf(fr, hf, dt):

    # pad frequency axis from 0 to f_min
    fmin    = np.min(fr)
    df      = fr[1]-fr[0]
    num     = int(fmin//df)
    if num > 0 :
        fr      = np.concatenate([np.arange(num)*df, fr])
        hf      = np.concatenate([np.zeros(num, dtype=complex), hf])

    # pad frequency axis from f_max to f_Nyq
    fnyq    = 0.5/dt
    fmax    = np.max(fr)
    if fnyq-fmax >= df:
        num = int((fnyq-fmax)//df)
        if num > 0 :
            fr  = np.concatenate([fr, np.arange(num)*df+fmax+df])
            hf  = np.concatenate([hf, np.zeros(num, dtype=complex)])

    # compute ifft
    time, ht = ifft(hf, 1./dt, 1./df)
    return ht

def read_gwosc(ifo, GPSstart, GPSend, srate=4096, version=None):
    """
        Read GW OpenScience in order to fetch the data,
        this method uses gwpy
    """

    from gwpy.timeseries import TimeSeries
    data    = TimeSeries.fetch_open_data(ifo, GPSstart, GPSend,
                                         sample_rate=srate,
                                         version=version,
                                         verbose=True,
                                         tag='CLN',
                                         format='hdf5',
                                         host='https://www.gw-openscience.org')

    s   = np.array(data.value)
    t   = np.arange(len(s))*(1./srate) + GPSstart
    return t , s

def read_data(data_flag, data_path, srate):

    if data_flag == 'inject' or data_flag == 'local' or  data_flag == 'gwosc':

        time, data = np.genfromtxt(data_path, unpack=True)
        srate_dt = 1./float(time[1] - time[0])
        if (srate > srate_dt):
            raise ValueError("You requested a sampling rate higher than the data sampling.")
        elif (srate < srate_dt):
            sys.stdout.write('Requested sampling rate is lower than data sampling rate. Downsampling detector data from {} to {} Hz, decimate factor {}\n'.format(srate_dt, srate, int(srate_dt/srate)))
            data = decimate(data, int(srate_dt/srate), zero_phase=True)
        else:
            pass
    else:
        raise KeyError("Please specify a data_flag.")
    return data

def read_asd(asd_path, ifo):
    from .. import __known_events__
    if asd_path == 'design':
        return get_design_sensitivity(ifo)
    elif asd_path in __known_events__:
        return get_event_sensitivity(ifo,asd_path)
    else:
        asd_path = os.path.abspath(asd_path)
        return np.genfromtxt(asd_path , usecols=[0,1], unpack=True)

def read_spcal(spcal_path, ifo):
    from .. import __known_events__
    if spcal_path in __known_events__:
        return get_event_calibration(ifo,spcal_path)
    else:
        f, mag, phi, sigma_mag_low, sigma_phi_low, sigma_mag_up, sigma_phi_up  = np.transpose(np.genfromtxt(spcal_path , usecols=[0,1,2,3,4,5,6]))
        amp_sigma = (sigma_mag_up - sigma_mag_low)/2.
        phi_sigma = (sigma_phi_up - sigma_phi_low)/2.
        return [f,amp_sigma,phi_sigma]

def read_params(path, flag):

    try:
        import configparser
    except ImportError:
        import ConfigParser as configparser

    config = configparser.ConfigParser()
    config.optionxform = str
    config.sections()
    config.read(path)

    params_list = list(config[flag].keys())
    params = {}
    for ki in params_list:
        if ki == 'approx':
            params[ki] = config[flag][ki]
        elif ki == 'lmax':
            params[ki] = np.int(config[flag][ki])
        elif ki == 'Eprior':
            params[ki] = config[flag][ki]
        elif ki == 'nqc-TEOBHyp':
            params[ki] = int(config[flag][ki])
        else:
            params[ki] = np.float(config[flag][ki])

    if 's1x' not in params_list:
        params['s1x'] = 0.
    if 's1y' not in params_list:
        params['s1y'] = 0.
    if 's1z' not in params_list:
        params['s1z'] = 0.
    if 's2x' not in params_list:
        params['s2x'] = 0.
    if 's2y' not in params_list:
        params['s2y'] = 0.
    if 's2z' not in params_list:
        params['s2z'] = 0.

    if 'lambda1' not in params_list:
        params['lambda1'] = 0.
    if 'lambda2' not in params_list:
        params['lambda2'] = 0.

    if 'cosi' not in params_list and 'iota' not in params_list:
        params['cosi'] = 1.
        params['iota'] = 0.

    if 'ra' not in params_list:
        params['ra'] = 0.
    if 'dec' not in params_list:
        params['dec'] = 0.

    if 'psi' not in params_list:
        params['psi'] = 0.
    if 'phi_ref' not in params_list:
        params['phi_ref'] = 0.
    if 'time_shift' not in params_list:
        params['time_shift'] = 0.

    if 'eccentricity' not in params_list:
        params['eccentricity'] = 0.

    if 'lmax' not in params_list:
        params['lmax'] = 0.
        
    if 'Eprior' not in params_list:
        params['Eprior'] = 'Constrained'
    if 'nqc-TEOBHyp' not in params_list:
        params['nqc-TEOBHyp'] = 1

    return params
