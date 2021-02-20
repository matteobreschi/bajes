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
            # shorter waveform: add tail, fill with zeros
            # tailing, if needed
            if hp[0] != 0 or hc[0] != 0:
                hp, hc  = tailing(hp, hc, srate, min(int(lenFin*alpha_tukey), lenFin-len(h)))
            # filling with zeros
            ldiff   = lenFin-len(hp)
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

        self.f_max  = np.max(self.freqs)
        self.df     = 1./self.seglen
        self.dt     = 1./self.srate

        self.times  = np.arange(int(self.srate*self.seglen))*self.dt

        self.approx = approx
        logger.info("Setting {} waveform ...".format(self.approx))

        # List all the approximants and assign functions for
        # hplus and hcross components in frequency-domain.
        # This function should take two arguments : self.freqs, params
        # Include here initialization commands if necessary.

        if self.approx == 'TaylorF2_3.5PN':
            from .approx.taylorf2 import taylorf2_35pn_wrapper
            self.wave_func  = taylorf2_35pn_wrapper
            self.domain     = 'freq'

        elif self.approx == 'TaylorF2_5.5PN':
            from .approx.taylorf2 import taylorf2_55pn_wrapper
            self.wave_func = taylorf2_55pn_wrapper
            self.domain     = 'freq'

        elif self.approx == 'TaylorF2_5.5PN_7.5PNTides':
            from .approx.taylorf2 import taylorf2_55pn75pntides_wrapper
            self.wave_func = taylorf2_55pn75pntides_wrapper
            self.domain     = 'freq'

        elif self.approx == 'TaylorF2_5.5PN_3.5PNQM_7.5PNTides':
            from .approx.taylorf2 import taylorf2_55pn35pnqm75pntides_wrapper
            self.wave_func = taylorf2_55pn35pnqm75pntides_wrapper
            self.domain     = 'freq'

        elif self.approx == 'TaylorF2_5.5PN_7.5PNTides2020':
            from .approx.taylorf2 import taylorf2_55pn75pnnewtides_wrapper
            self.wave_func = taylorf2_55pn75pnnewtides_wrapper
            self.domain     = 'freq'

        elif self.approx == 'TEOBResumS':
            from .approx.teobresums import teobresums_wrapper
            self.wave_func = teobresums_wrapper
            self.domain     = 'time'

        elif self.approx == 'TEOBResumSPA':
            from .approx.teobresums import teobresums_spa_wrapper
            self.wave_func  = teobresums_spa_wrapper
            self.domain     = 'freq'

        elif self.approx == 'TEOBResumSPA_NRPMw':
            from .approx.teobresums import teobresums_spa_nrpmw_wrapper
            self.wave_func  = teobresums_spa_nrpmw_wrapper
            self.domain     = 'freq'

        elif self.approx == 'HypTEOBResumS':
            from .approx.teobresums import teobresums_hyperb_wrapper
            self.wave_func  = teobresums_hyperb_wrapper
            self.domain     = 'time'

        elif self.approx == 'NRPM':
            from .approx.nrpm import nrpm_wrapper
            self.wave_func  = nrpm_wrapper
            self.domain     = 'time'

        elif self.approx == 'NRPM_ext':
            from .approx.nrpm import nrpm_extended_wrapper
            self.wave_func  = nrpm_extended_wrapper
            self.domain     = 'time'

        elif self.approx == 'NRPM_ext_recal':
            from .approx.nrpm import nrpm_extended_recal_wrapper
            self.wave_func  = nrpm_extended_recal_wrapper
            self.domain     = 'time'

        elif self.approx == 'NRPMw':
            from .approx.nrpmw import nrpmw_wrapper
            self.wave_func  = nrpmw_wrapper
            self.domain     = 'freq'

        elif self.approx == 'NRPMw_recal':
            from .approx.nrpmw import nrpmw_recal_wrapper
            self.wave_func  = nrpmw_recal_wrapper
            self.domain     = 'freq'

        elif self.approx == 'TEOBResumS_NRPM':
            from .approx.teobresums import teobresums_nrpm_wrapper
            self.wave_func  = teobresums_nrpm_wrapper
            self.domain     = 'time'

        elif self.approx == 'NRSur7dq4':
            # OBS: GWSurrogate functions are class with a __call__ method,
            # instead of standard methods, because it has to download the catalog
            from .approx.gwsurrogate import nrsur7dq4_wrapper
            self.wave_func  = nrsur7dq4_wrapper()
            self.domain     = 'time'

        elif self.approx == 'NRHybSur3dq8':
            # OBS: GWSurrogate functions are class with a __call__ method,
            # instead of standard methods, because it has to download the catalog
            from .approx.gwsurrogate import nrhybsur3dq8_wrapper
            self.wave_func  = nrhybsur3dq8_wrapper()
            self.domain     = 'time'

        elif self.approx == 'NRHybSur3dq8Tidal':
            # OBS: GWSurrogate functions are class with a __call__ method,
            # instead of standard methods, because it has to download the catalog
            from .approx.gwsurrogate import nrhybsur3dq8tidal_wrapper
            self.wave_func = nrhybsur3dq8tidal_wrapper()
            self.domain = 'time'

        elif self.approx == 'MLGW':
            # OBS: MLGW functions are class with a __call__ method,
            # instead of standard methods, because it has to initilize the generator
            from .approx.mlgw import mlgw_wrapper
            self.wave_func = mlgw_wrapper(self.seglen, self.srate)
            self.domain = 'time'

        elif self.approx == 'MLGW-BNS':
            # OBS: MLGW-BNS functions are class with a __call__ method,
            # instead of standard methods, because it has to initilize the generator
            from .approx.mlgw import mlgw_bns_wrapper
            self.wave_func = mlgw_bns_wrapper(self.freqs, self.seglen, self.srate)
            self.domain = 'freq'

        elif self.approx == 'MLTEOBNQC':
            # OBS: MLGW functions are class with a __call__ method,
            # instead of standard methods, because it has to initilize the generator
            from .approx.mlgw import mlteobnqc_wrapper
            self.wave_func = mlteobnqc_wrapper(self.seglen, self.srate)
            self.domain = 'time'

        elif self.approx == 'MLSEOBv4':
            # OBS: MLGW functions are class with a __call__ method,
            # instead of standard methods, because it has to initilize the generator
            from .approx.mlgw import mlseobv4_wrapper
            self.wave_func = mlseobv4_wrapper(self.seglen, self.srate)
            self.domain = 'time'

        elif 'LALSimFD' in self.approx:
            # OBS: LALSim functions are class with a __call__ method,
            # instead of standard methods
            from .approx.lal import lal_wrapper
            lal_approx      = self.approx.split('-')[1]
            self.domain     = 'freq'
            self.wave_func  = lal_wrapper(lal_approx, self.domain)

        elif 'LALSimTD' in self.approx:
            # OBS: LALSim functions are class with a __call__ method,
            # instead of standard methods
            from .approx.lal import lal_wrapper
            lal_approx      = self.approx.split('-')[1]
            self.domain     = 'time'
            self.wave_func  = lal_wrapper(lal_approx, self.domain)

        else:
            from . import __known_approxs__
            raise AttributeError("Unable to read approximant string. Please use a valid value (see bajes.__known_approxs__):\n{}.\nIf you are using a LAL approximant, it is possible to know the full list at https://lscsoft.docs.ligo.org/lalsuite/".format(__known_approxs__))

    def compute_hphc(self, params):
        """ Compute waveform for compact binary coalescences
            --------
            params : dictionary
            Dictionary of the parameters, with the following

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
                distance    : luminosiry distance [Mpc]
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

        elif self.domain == 'freq':
            # time-shifting (amplitude peak as central value)
            hp = hp * np.exp(1j*np.pi*self.freqs*self.seglen)
            hc = hc * np.exp(1j*np.pi*self.freqs*self.seglen)

        return PolarizationTuple(plus=hp, cross=hc)
