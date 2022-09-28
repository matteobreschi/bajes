from __future__ import division, unicode_literals, absolute_import
import numpy as np

import logging
logger = logging.getLogger(__name__)

from ...pipe import cart2sph , sph2cart
from .strain import windowing, lagging

from .utils import compute_lambda_tilde, compute_delta_lambda
from ..utils.tov import TOVSolver

from collections import namedtuple
PolarizationTuple = namedtuple("PolarizationTuple", ("plus","cross"), defaults=([None],[None]))
# TODO : introduce scalar, vector, tensor polarizations for polarization tests

# # Dictionary of known approximants
# # Each key corresponds to the name of the approximant
# # Each value has to be a dictionary
# # that include the following keys:
# #   * 'path':   string to method to be imported, e.g. bajes.obs.gw.approx.taylorf2.taylorf2_35pn_wrapper
# #   * 'type':   define if the passed func is a function or a class, options: ['fnc', 'cls']
# #   * 'domain': define if the method returns a frequency- or time-domain waveform, options: ['time', 'freq']

__approx_dict__ = { ### TIME-DOMAIN
                    # funcs
                    'TEOBResumS':                           {'path': 'bajes.obs.gw.approx.teobresums.teobresums_wrapper',
                                                             'type': 'fnc',
                                                             'domain': 'time'},
                    'TEOBResumS_NRPM':                      {'path': 'bajes.obs.gw.approx.teobresums.teobresums_nrpm_wrapper',
                                                             'type': 'fnc',
                                                             'domain': 'time'},
                    'EccTEOBResumS':                        {'path': 'bajes.obs.gw.approx.teobresums.teobresums_ecc_wrapper',
                                                             'type': 'fnc',
                                                             'domain': 'time'},
                    'HypTEOBResumS':                        {'path': 'bajes.obs.gw.approx.teobresums.teobresums_hyperb_wrapper',
                                                             'type': 'fnc',
                                                             'domain': 'time'},
                    'NRPM':                                 {'path': 'bajes.obs.gw.approx.nrpm.nrpm_wrapper',
                                                             'type': 'fnc',
                                                             'domain': 'time'},
                    'NRPM_ext':                             {'path': 'bajes.obs.gw.approx.nrpm.nrpm_extended_wrapper',
                                                             'type': 'fnc',
                                                             'domain': 'time'},
                    'NRPM_ext_recal':                       {'path': 'bajes.obs.gw.approx.nrpm.nrpm_extended_recal_wrapper',
                                                             'type': 'fnc',
                                                             'domain': 'time'},
                    # classes
                    'NRSur7dq4':                            {'path': 'bajes.obs.gw.approx.gwsurrogate.nrsur7dq4_wrapper',
                                                             'type': 'cls',
                                                             'domain': 'time'},
                    'NRHybSur3dq8':                         {'path': 'bajes.obs.gw.approx.gwsurrogate.nrhybsur3dq8_wrapper',
                                                             'type': 'cls',
                                                             'domain': 'time'},
                    'NRHybSur3dq8Tidal':                    {'path': 'bajes.obs.gw.approx.gwsurrogate.nrhybsur3dq8tidal_wrapper',
                                                             'type': 'cls',
                                                             'domain': 'time'},
                    'MLGW':                                 {'path': 'bajes.obs.gw.approx.mlgw.mlgw_wrapper',
                                                             'type': 'cls',
                                                             'domain': 'time'},
                    'MLTEOBNQC':                            {'path': 'bajes.obs.gw.approx.mlgw.mlteobnqc_wrapper',
                                                             'type': 'cls',
                                                             'domain': 'time'},
                    'MLSEOBv4':                             {'path': 'bajes.obs.gw.approx.mlgw.mlseobv4_wrapper',
                                                             'type': 'cls',
                                                             'domain': 'time'},
                    'LALSimTD':                             {'path': 'bajes.obs.gw.approx.lal.lal_wrapper',
                                                             'type': 'cls',
                                                             'domain': 'time'},
                    ### FREQUENCY-DOMAIN
                    # funcs
                    'TaylorF2_3.5PN':                       {'path':   'bajes.obs.gw.approx.taylorf2.taylorf2_35pn_wrapper',
                                                             'type':   'fnc',
                                                             'domain': 'freq'},
                    'TaylorF2_5.5PN':                       {'path': 'bajes.obs.gw.approx.taylorf2.taylorf2_55pn_wrapper',
                                                             'type': 'fnc',
                                                             'domain': 'freq'},
                    'TaylorF2_5.5PN_7.5PNTides':            {'path': 'bajes.obs.gw.approx.taylorf2.taylorf2_55pn75pntides_wrapper',
                                                             'type': 'fnc',
                                                             'domain': 'freq'},
                    'TaylorF2_5.5PN_3.5PNQM_7.5PNTides':    {'path': 'bajes.obs.gw.approx.taylorf2.taylorf2_55pn35pnqm75pntides_wrapper',
                                                             'type': 'fnc',
                                                             'domain': 'freq'},
                    'TaylorF2_5.5PN_7.5PNTides2020':        {'path': 'bajes.obs.gw.approx.taylorf2.taylorf2_55pn75pnnewtides_wrapper',
                                                             'type': 'fnc',
                                                             'domain': 'freq'},
                    'TEOBResumSPA':                         {'path': 'bajes.obs.gw.approx.teobresums.teobresums_spa_wrapper',
                                                             'type': 'fnc',
                                                             'domain': 'freq'},
                    'TEOBResumSPA_NRPMw':                   {'path': 'bajes.obs.gw.approx.teobresums.teobresums_spa_nrpmw_wrapper',
                                                             'type': 'fnc',
                                                             'domain': 'freq'},
                    'TEOBResumSPA_NRPMw_recal':             {'path': 'bajes.obs.gw.approx.teobresums.teobresums_spa_nrpmw_recal_wrapper',
                                                             'type': 'fnc',
                                                             'domain': 'freq'},
                    'NRPMw':                                {'path': 'bajes.obs.gw.approx.nrpmw.nrpmw_wrapper',
                                                             'type': 'fnc',
                                                             'domain': 'freq'},
                    'NRPMw_recal':                          {'path': 'bajes.obs.gw.approx.nrpmw.nrpmw_recal_wrapper',
                                                             'type': 'fnc',
                                                             'domain': 'freq'},
                    # classes
                    'MLGW-BNS':                             {'path': 'bajes.obs.gw.approx.mlgw.mlgw_bns_wrapper',
                                                             'type': 'cls',
                                                             'domain': 'freq'},
                    'MLGW-BNS_NRPMw':                       {'path': 'bajes.obs.gw.approx.mlgw.mlgw_bns_nrpmw_wrapper',
                                                             'type': 'cls',
                                                             'domain': 'freq'},
                    'MLGW-BNS_NRPMw_recal':                 {'path': 'bajes.obs.gw.approx.mlgw.mlgw_bns_nrpmw_recal_wrapper',
                                                             'type': 'cls',
                                                             'domain': 'freq'},
                    'LALSimFD':                             {'path': 'bajes.obs.gw.approx.lal.lal_wrapper',
                                                             'type': 'cls',
                                                             'domain': 'freq'}
                  }

# TODO: describe waveform generator structure
def __get_waveform_generator__(approx, seglen, srate):

    # get approximant list
    __known_approxs__ = list(__approx_dict__.keys())

    # switch between LAL approx and others
    if ('LALSimTD' not in approx) and ('LALSimFD' not in approx):

        # non-LAL waaveforms
        if (approx not in __known_approxs__):
            logger.error("Unable to read approximant string. Please use a valid string: {}.\nIf you are using a LAL approximant, the full list can be found at https://lscsoft.docs.ligo.org/lalsuite/".format(__known_approxs__))
            raise AttributeError("Unable to read approximant string. Please use a valid string: {}.\nIf you are using a LAL approximant, the full list can be found at https://lscsoft.docs.ligo.org/lalsuite/".format(__known_approxs__))

        this_wave = __approx_dict__[approx]
        wave_pars = {'seglen': seglen, 'srate': srate, 'domain': this_wave['domain']}

    else:
        # if enters here LALSimTD or LALSimFD is in approx name
        lal_approx  = approx.split('-')
        if (lal_approx[0]!='LALSimTD') or (lal_approx[0]!='LALSimFD'):
            logger.error("Wrong syntax for LAL approximant, please use: [ LALSimTD or LALSimFD ]-[ LAL approx name ].\nThe full list of LAL approximants can be found at https://lscsoft.docs.ligo.org/lalsuite/")
            raise AttributeError("Wrong syntax for LAL approximant, please use: [ LALSimTD or LALSimFD ]-[ LAL approx name ].\nThe full list of LAL approximants can be found at https://lscsoft.docs.ligo.org/lalsuite/")

        this_wave = __approx_dict__[lal_approx[0]]
        wave_pars = {'seglen': seglen, 'srate': srate, 'domain': this_wave['domain'], 'approx': lal_approx[1]}

    # set module string and import
    from importlib import import_module
    path_to_method  = this_wave['path'].split('.')
    wave_module     = import_module('.'.join(path_to_method[:-1]))

    # this condition never occurs if the code is properly written
    if path_to_method[-1] not in dir(wave_module):
        raise AttributeError("Unable to import {} method from {}".format(path_to_method[-1], wave_module))

    # get waveform generator and domain string
    if this_wave['type'] == 'fnc':
        wave_func = getattr(wave_module, path_to_method[-1])
        wave_domn = this_wave['domain']
    elif this_wave['type'] == 'cls':
        wave_obj  = getattr(wave_module, path_to_method[-1])
        wave_func = wave_obj(**wave_pars)
        wave_domn = this_wave['domain']
    else:
        # this condition never occurs if the __approx_dict__ is properly written
        raise AttributeError("Unable to define method of type {} for waveform generator. Check bajes.obs.gw.waveform.__approx_dict__".format(wave_pars['type']))

    return wave_func, wave_domn

def tailing(hp, hc, srate, npt):

    # estimate initial properties
    dt      = 1./srate
    h       = hp - 1j*hc
    phi     = np.unwrap(np.angle(h))
    amp     = np.abs(h)
    a0      = np.abs(amp)[0]
    p0      = phi[0]
    f0      = np.abs(phi[2]-phi[1])/dt/(2*np.pi)

    # generate tail and attach
    t       = np.arange(npt)*dt
    amp_t   = a0*0.5*(1. - np.cos(np.pi*t/np.max(t)))
    phi_t   = -2*np.pi*(t-np.max(t)-dt)*f0 + p0
    tail    = amp_t * np.exp( 1j*phi_t )
    h       = np.concatenate([tail, h])

    return np.real(h), -np.imag(h)

def centering_tdwave(hp, hc, seglen, srate, alpha_tukey = 0.1):

    h           = hp - 1j*hc
    imax        = np.argmax(np.abs(h))
    lenFin      = int(srate*seglen)

    if len(h) == lenFin and imax == lenFin//2:
        # if waveform is already centered with the correct length, return input + window
        nlag = 0
    elif len(h) == lenFin:
        # if waveform has the correct length, return centering + window
        nlag = lenFin//2 - imax
    else:
        if len(h) < lenFin:
            ### NOTE: as it is now, the tailing function can introduce wrong frequencies for eccentric, precessing and/or HOM waveforms
            # shorter waveform: add tail, fill with zeros
            # # tailing, if needed
            # if hp[0] != 0 or hc[0] != 0:
            #     hp, hc  = tailing(hp, hc, srate, min(int(lenFin*alpha_tukey), lenFin-len(h)))
            # filling with zeros
            ldiff   = lenFin-len(h)
            hp, hc  = np.append(np.zeros(ldiff), hp), np.append(np.zeros(ldiff), hc)
            # centering
            imax    = np.argmax(np.abs(hp - 1j*hc))
            nlag    = lenFin//2 - imax
        else:
            # longer waveform: cut tail
            # cutting
            ldiff   = len(hp)-lenFin
            hp, hc  = hp[ldiff:], hc[ldiff:]
            # centering
            imax    -= ldiff
            nlag    = lenFin//2 - imax

    # lagging + windowing
    hp, wfact = windowing(lagging(hp,nlag), alpha_tukey)
    hc, wfact = windowing(lagging(hc,nlag), alpha_tukey)

    return hp, hc

class Waveform(object):
    """
        Waveform model object for compact binary coalescences
    """

    def __init__(self, freqs, srate, seglen, approx):
        """
            Initialize the Waveform with a frequency axis and the name of the approximant
        """

        self.freqs  = freqs
        self.seglen = seglen
        self.srate  = srate

        self.f_min  = np.amin(self.freqs)
        self.f_max  = np.amax(self.freqs)
        self.df     = 1./self.seglen
        self.dt     = 1./self.srate

        self.times  = np.arange(int(self.srate*self.seglen))*self.dt

        self.approx = approx
        logger.info("Setting {} waveform ...".format(self.approx))

        # get waveform generator from string
        self.wave_func, self.domain = __get_waveform_generator__(self.approx, self.seglen, self.srate)


    def compute_hphc(self, params, roq=None):
        """ Compute waveform for compact binary coalescences
            --------
            params : dictionary
            Dictionary of the parameters, with the following:

                [masses notation n.1 ]
                mchirp      : chirp mass [solar masses]
                q           : mass ratio [ > 1 ]

                [spins notation n.1 - cartesian]
                s1x         : primary spin component along x axis [dimensionless]
                s1y         : primary spin component along y axis [dimensionless]
                s1z         : primary spin component along z axis [dimensionless]
                s2x         : secondary spin component along x axis [dimensionless]
                s2y         : secondary spin component along y axis [dimensionless]
                s2z         : secondary spin component along z axis [dimensionless]

                [spins notation n.2 - spherical]
                s1          : primary spin magnitude [dimensionless]
                tilt1       : azimuathal angle primary spin [rad]
                phi_1l      : equatorial angle primary spin [rad]
                s2          : secondary spin magnitude [dimensionless]
                tilt2       : azimuathal angle secondary spin [rad]
                phi_2l      : equatorial angle secondary spin [rad]

                [tides]
                lambda1     : primary tidal component [dimensionless]
                lambda2     : secondary tidal component [dimensionless]

                [extrinsic]
                distance    : luminosity distance [Mpc]
                iota        : inclination angle [rad]
                cosi        : cos(iota), alternative to iota
                phi_ref     : reference phase [rad]

                [others]
                (these values may be used from the waveform generator,
                depending on the selected approximant)
                f_min       : initial frequency [Hz]
                f_max       : maximum frequency [Hz]
                srate       : sampling rate [Hz]
                seglen      : duration of the segment [s]

                [used to project on the detector]
                (in general these values are not needed,
                but they are mandatory if you want to call
                project_fdwave or project_tdwave methods)
                ra          : right ascension [rad]
                dec         : declination [rad]
                psi         : polarization angle [rad]
                time_shift  : time-shift from t-gps to t-ref at Earth's geocenter [sec]
                t_gps       : GPS trigger time

            roq : dictionary
            Dictionary containing ROQ options, used to skip time-shifting.

            --------
            return hp, hc
        """

        # parse masses
        # ensure at least  [mchirp, mtot, q] in params
        # obs. the sampler works with (mchirp,q) or with (mtot,q)
        if 'mchirp' in params.keys():
            nu              = params['q']/(1.+params['q'])**2.
            params['mtot']  = params['mchirp']/nu**0.6
        elif 'mtot' in params.keys():
            nu                  = params['q']/(1.+params['q'])**2.
            params['mchirp']    = params['mtot']*nu**0.6

        # parse primary spin
        if 'tilt1' in params.keys() and 's1' in params.keys() and 'phi_1l' in params.keys():
            params['s1x'], params['s1y'], params['s1z'] = sph2cart(params['s1'], params['tilt1'], params['phi_1l'])

        # parse secondary spin
        if 'tilt2' in params.keys() and 's2' in params.keys() and 'phi_2l' in params.keys():
            params['s2x'], params['s2y'], params['s2z'] = sph2cart(params['s2'], params['tilt2'], params['phi_2l'])

        # parse for TOV solver
        if 'eos_logp1' in params.keys():

            logger.debug("Solving EOS for [{}, {}, {}, {}]".format(params['eos_logp1'], params['eos_gamma1'], params['eos_gamma2'], params['eos_gamma3']))

            # initialize EOS solver
            tov = TOVSolver([params['eos_logp1'], params['eos_gamma1'], params['eos_gamma2'], params['eos_gamma3']])

            if not tov.is_physical:
                return PolarizationTuple()

            # compute tidal deformabilities
            if 'lambda1' not in params.keys():
                m1  = params['mtot']*params['q']/(1.+params['q'])
                if m1 > tov.Mmax: return PolarizationTuple()
                params['lambda1'] = tov.tidal_deformability(m1)
                logger.debug("Estimated lambda={} for mass={}".format(params['lambda1'], m1))

            if 'lambda2' not in params.keys():
                m2  = params['mtot']/(1.+params['q'])
                if m2 > tov.Mmax: return PolarizationTuple()
                params['lambda2'] = tov.tidal_deformability(m2)
                logger.debug("Estimated lambda={} for mass={}".format(params['lambda2'], m2))

        # include iota
        if 'cosi' in params.keys():
            params['iota'] = np.arccos(params['cosi'])
        
        # compute hplus and hcross according with approx
        hp , hc = self.wave_func(self.freqs, params)

        # if hp,hc is None, return Nones
        if not any(hp):
            return PolarizationTuple()

        if self.domain == 'time':
            # padding + time-shifting (amplitude peak as central value)
            hp, hc = centering_tdwave(hp, hc, self.seglen, self.srate, params['tukey'])

        elif((self.domain == 'freq') and (roq==None)):
            # Time-shifting the amplitude peak to the center of the data segment (seglen/2). In the ROQ case, this is done separately.
            hp = hp * np.exp(1j*np.pi*self.freqs*self.seglen)
            hc = hc * np.exp(1j*np.pi*self.freqs*self.seglen)

        return PolarizationTuple(plus=hp, cross=hc)
