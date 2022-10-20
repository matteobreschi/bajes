from __future__ import division, unicode_literals, absolute_import
import numpy as np
import os

import logging, warnings
logger = logging.getLogger(__name__)

__url__ = 'https://pypi.org/project/gwsurrogate/'

try:
    import gwsurrogate
except ImportError:
    warnings.warn("Unable to import GWSurrogate package. Please see related documentation at: {}".format(__url__))
    logger.warning("Unable to import GWSurrogate package. Please see related documentation at: {}".format(__url__))
    pass

class nrsur7dq4_wrapper(object):

    def __init__(self, **kwargs):

        if not os.path.exists(gwsurrogate.__path__[0]+'/surrogate_downloads/NRSur7dq4.h5'):
            gwsurrogate.catalog.pull('NRSur7dq4')

        self.sur    = gwsurrogate.LoadSurrogate('NRSur7dq4')

    def __call__(self, freqs, params):
        hp, hc, dt      = self.td_waveform(params)
        return hp, hc

    def td_waveform(self, params):

        nrsurr_params = {}
        nrsurr_params['dt']     = 1./params['srate']
        nrsurr_params['f_low']  = 0.

        chiA   = [params['s1x'],params['s1y'],params['s1z']]
        chiB   = [params['s2x'],params['s2y'],params['s2z']]

        if params['lmax'] < 2:
            nrsurr_params['mode_list'] = [(2,2)]
        else:
            nrsurr_params['ellMax'] = params['lmax']

        t, h, dyn = self.sur(params['q'], chiA, chiB,
                             # if f_low, start from the minimum freq of the surrogate
                             # without f_ref, f_ref = f_low
                             M          = params['mtot'],
                             dist_mpc   = params['distance'],
                             inclination = params['iota'],
                             phi_ref    = params['phi_ref'],
                             tidal_opts = {'Lambda1': params['lambda1'],
                                           'Lambda2': params['lambda2']},
                             units      = 'mks',
                             skip_param_checks = True,
                             **nrsurr_params)

        hp = h.real
        hc = h.imag
        return hp, hc, nrsurr_params['dt']

class nrhybsur3dq8_wrapper(object):

    def __init__(self, **kwargs):

        if not os.path.exists(gwsurrogate.__path__[0]+'/surrogate_downloads/NRHybSur3dq8.h5'):
            gwsurrogate.catalog.pull('NRHybSur3dq8')

        self.sur    = gwsurrogate.LoadSurrogate('NRHybSur3dq8')

    def __call__(self, freqs, params):
        hp, hc, dt      = self.td_waveform(params)
        return hp, hc

    def td_waveform(self, params):
        srate = params['srate']
        chiA = [params['s1x'],params['s1y'],params['s1z']]
        chiB = [params['s2x'],params['s2y'],params['s2z']]

        if params['lmax'] < 2:
            params['lmax'] = 2

        t, h, dyn = self.sur(params['q'], chiA, chiB,
                             dt         = 1./srate,
                             # if f_low, start from the minimum freq of the surrogate
                             # without f_ref, f_ref = f_low
                             f_low      = 0,
                             ellMax     = params['lmax'],
                             M          = params['mtot'],
                             dist_mpc   = params['distance'],
                             inclination = params['iota'],
                             phi_ref    = params['phi_ref'],
                             tidal_opts = {'Lambda1': 0.,
                                           'Lambda2': 0.},
                             units      = 'mks',
                             skip_param_checks = True)

        hp = h.real
        hc = h.imag
        dt = np.median(np.diff(t))
        return hp, hc, dt

class nrhybsur3dq8tidal_wrapper(object):

    def __init__(self, **kwargs):

        if not os.path.exists(gwsurrogate.__path__[0]+'/surrogate_downloads/NRHybSur3dq8.h5'):
            gwsurrogate.catalog.pull('NRHybSur3dq8')

        self.sur    = gwsurrogate.LoadSurrogate('NRHybSur3dq8Tidal')

    def __call__(self, freqs, params):
        hp, hc, dt      = self.td_waveform(params)
        return hp, hc

    def td_waveform(self, params):

        srate = params['srate']
        chiA = [params['s1x'],params['s1y'],params['s1z']]
        chiB = [params['s2x'],params['s2y'],params['s2z']]

        if params['lmax'] < 2:
            params['lmax'] = 2

        t, h, dyn = self.sur(params['q'], chiA, chiB,
                             dt         = 1./srate,
                             # if f_low, start from the minimum freq of the surrogate
                             # without f_ref, f_ref = f_low
                             f_low      = 0,
                             ellMax     = params['lmax'],
                             M          = params['mtot'],
                             dist_mpc   = params['distance'],
                             inclination = params['iota'],
                             phi_ref    = params['phi_ref'],
                             tidal_opts = {'Lambda1': params['lambda1'],
                                           'Lambda2': params['lambda1']},
                             units      = 'mks',
                             skip_param_checks = True)

        hp = h.real
        hc = h.imag
        dt = np.median(np.diff(t))
        return hp, hc, dt
